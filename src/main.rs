use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::collections::HashSet;
use arrow_schema::{Field, Schema, DataType};
use arrow_array::{RecordBatch, StringArray, RecordBatchIterator, Array};
use arrow_array::builder::{PrimitiveBuilder, FixedSizeListBuilder};
use arrow_array::types::Float32Type;
use lancedb::query::{ExecutableQuery, QueryBase};
use datafusion::prelude::*;

#[derive(Parser)]
#[command(name = "sap-to-ai-bridge")]
#[command(about = "Bridging SAP ERP data with Semantic AI Search", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a CSV file from SAP into the Vector Database
    Ingest {
        /// The path to the CSV file (e.g. data/kna1.csv)
        #[arg(short, long)]
        file: String,
        /// Overwrite the existing database instead of appending
        #[arg(short, long)]
        overwrite: bool,
    },
    /// Query the Vector Database using natural language
    Ask {
        /// The semantic question (e.g. "Find customers in Germany")
        query: String,
    },
    /// Execute a raw SQL query against the SAP data (Zero-ETL)
    AskSql {
        /// The SQL query to execute
        query: String,
    },
    /// AI-Generated Hybrid Query (Dynamically routes to SQL or Semantic Search)
    AskAiSql {
        /// The natural language question
        query: String,
    },
}

fn download_file(url: &str, dest: &str) -> Result<String, Box<dyn Error>> {
    if !Path::new(dest).exists() {
        println!("Downloading {}...", dest);
        let resp = ureq::get(url).call()?;
        let mut out = File::create(dest)?;
        std::io::copy(&mut resp.into_reader(), &mut out)?;
    }
    Ok(dest.to_string())
}

/// Setup Phase: This function runs exactly once at startup.
/// It downloads (if necessary) and loads the AI model and tokenizer into memory.
/// These instantiated objects are then kept alive and used to vectorize every piece of SAP data we feed them.
fn load_model() -> Result<(BertModel, Tokenizer), Box<dyn Error>> {
    println!("Fetching safetensors and config...");

    // Grab the absolute path to the project root (where Cargo.toml is)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let data_dir = format!("{}/data", manifest_dir);

    // Ensure the absolute data directory exists
    std::fs::create_dir_all(&data_dir)?;

    // Download the 3 required files directly to the absolute data folder
    let config_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json",
        &format!("{}/config.json", data_dir),
    )?;
    let tokenizer_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/tokenizer.json",
        &format!("{}/tokenizer.json", data_dir),
    )?;
    let weights_path = download_file(
        "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/model.safetensors",
        &format!("{}/model.safetensors", data_dir),
    )?;

    println!("Building Tokenizer and Tensor Neural Network...");

    // Load the model configuration from JSON.
    let config = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config)?;

    // Initialize the tokenizer.
    let mut tokenizer =
        Tokenizer::from_file(tokenizer_path).map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    // Configure the tokenizer with batch longest padding.
    tokenizer.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    }));

    // Map the safetensors model weights into memory.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &Device::Cpu)?
    };

    // Load the BERT model weights.
    let model = BertModel::load(vb, &config)?;

    Ok((model, tokenizer))
}

// Normalize a vector to unit length using L2 normalization
fn normalize_vector(mut vec: Vec<f32>) -> Vec<f32> {
    let magnitude = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if magnitude > 0.0 {
        vec.iter_mut().for_each(|x| *x /= magnitude);
    }
    vec
}

fn get_embeddings(
    sentences: &[String],
    tokenizer: &Tokenizer,
    model: &BertModel,
) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    // 1. Tokenize input sentences
    let inputs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let tokens = tokenizer
        .encode_batch(inputs, true)
        .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    // 2. Build token ID tensor
    let token_ids = tokens
        .iter()
        .map(|t| Tensor::new(t.get_ids(), &Device::Cpu))
        .collect::<Result<Vec<_>, _>>()?;
    let token_ids = Tensor::stack(&token_ids, 0)?;

    // 3. Forward pass through the model
    let token_type_ids = token_ids.zeros_like()?;
    let embeddings = model.forward(&token_ids, &token_type_ids, None)?;

    // 4. Apply CLS pooling to extract sentence-level embeddings
    let cls_embeddings = embeddings.i((.., 0, ..))?;

    // 5. Convert to Vec and apply L2 normalization
    let raw_vecs = cls_embeddings.to_vec2::<f32>()?;
    let normalized_vecs = raw_vecs.into_iter().map(normalize_vector).collect();

    Ok(normalized_vecs)
}
#[derive(Debug, Deserialize)]
struct Kna1Row {
    kunnr: Option<String>,
    name1: Option<String>,
    ort01: Option<String>,
    land1: Option<String>,
}

/// ==========================================
/// STEP 4: BLOCK 2 (LLM CLIENT)
/// ==========================================
/// Strict Rust struct mapping for the Ollama JSON request
#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

/// Strict Rust struct mapping for the Ollama JSON response
#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

/// Sends a prompt to the local Ollama server running Llama 3.2
async fn ask_llm(prompt: &str) -> Result<String, Box<dyn Error>> {
    let req_body = OllamaRequest {
        model: "llama3.2:latest".to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };

    let client = reqwest::Client::new();
    let res = client.post("http://localhost:11434/api/generate").json(&req_body).send().await?.json::<OllamaResponse>().await?;
    Ok(res.response)
}

/// ==========================================
/// STEP 4: BLOCK 3 (PROMPT ENGINEERING)
/// ==========================================
/// Builds the highly-constrained prompt to force the lightweight LLM 
/// into an exact JSON format using Few-Shot In-Context Learning.
fn build_routing_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and decide if it requires exact SQL or SEMANTIC search.

Database Schema for table `kna1`:
- kunnr (String): Customer Number / ID
- name1 (String): Customer Name
- ort01 (String): City
- land1 (String): Country Code (e.g., 'US', 'DE')

RULES:
1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.
2. The JSON must have two keys: \"route\" (either \"SQL\" or \"SEMANTIC\") and \"query\" (the generated SQL string, or blank).

Examples:
Q: \"How many customers are in Berlin?\"
A: {{\"route\": \"SQL\", \"query\": \"SELECT count(*) FROM kna1 WHERE ort01 = 'Berlin'\"}}

Q: \"Show me the names of 5 customers in the US.\"
A: {{\"route\": \"SQL\", \"query\": \"SELECT name1 FROM kna1 WHERE land1 = 'US' LIMIT 5\"}}

Q: \"Find customers who are large tech manufacturers.\"
A: {{\"route\": \"SEMANTIC\", \"query\": \"\"}}

User Question: \"{}\"
A: ",
        user_question
    )
}

/// Strict Rust struct to parse the LLM's dynamic JSON decision
#[derive(Deserialize, Debug)]
struct RouterDecision {
    route: String,
    query: String,
}

/// ==========================================
/// STEP 4: BLOCK 1 (APACHE DATAFUSION)
/// ==========================================
/// This function spins up an in-memory SQL engine using Apache DataFusion.
/// It registers our raw SAP CSV file as a SQL table (Zero-ETL) and runs the query.
async fn execute_sql_query(query: &str) -> Result<(), Box<dyn Error>> {
    println!("Spinning up Apache DataFusion Engine (Zero-Copy)...");
    
    // Create the execution context
    let ctx = SessionContext::new();
    
    // Register the CSV file as a virtual table named 'kna1'
    println!("Registering data/kna1.csv as virtual SQL table...");
    ctx.register_csv("kna1", "data/kna1.csv", CsvReadOptions::new()).await?;
    
    // Execute the raw SQL string
    println!("Executing SQL: {}", query);
    let df = ctx.sql(query).await?;
    
    // Print the formatted results to the terminal
    df.show().await?;
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse CLI arguments
    let cli = Cli::parse();

    match &cli.command {
        Commands::Ingest { file, overwrite } => {
            println!(">>> Executing INGEST command on file: {}", file);
            let csv_path = file.clone();

            println!("Connecting to LanceDB...");
            let db = lancedb::connect("data/sap_vectors").execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

            if *overwrite {
                println!("Overwrite flag detected. Dropping existing 'customers' table...");
                let _ = db.drop_table("customers").await;
            }

            // ==========================================
            // 1. Application-Level Deduplication
            // ==========================================
            // We query the DB for existing SAP keys (kunnr) BEFORE generating embeddings.
            // This saves massive CPU cycles by not running the neural network on data we already have.
            let mut existing_kunnrs = HashSet::new();
            if let Ok(table) = db.open_table("customers").execute().await {
                println!("Table exists. Scanning existing records to prevent redundant CPU embeddings...");
                use futures::StreamExt;
                if let Ok(mut stream) = table.query().execute().await {
                    while let Some(batch) = stream.next().await {
                        if let Ok(batch) = batch {
                            if let Some(col) = batch.column_by_name("kunnr") {
                                if let Some(kunnr_array) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                                    for i in 0..kunnr_array.len() {
                                        existing_kunnrs.insert(kunnr_array.value(i).to_string());
                                    }
                                } else {
                                    eprintln!("WARNING: 'kunnr' column is not a String. Skipping deduplication batch.");
                                }
                            } else {
                                eprintln!("WARNING: 'kunnr' column missing. Skipping deduplication batch.");
                            }
                        }
                    }
                }
            }

            println!("Reading {}...", csv_path);
            let file = File::open(&csv_path).map_err(|e| {
                format!(
                    "File load went wrong. Rust shows the following error: {}",
                    e
                )
            })?;

            let mut rdr = csv::Reader::from_reader(file);

            let mut documents = Vec::new();
            let mut records = Vec::new();

            // 1. Parse CSV and build the sentences
            for result in rdr.deserialize() {
                // Deserialize CSV row into the Kna1Row struct
                let record: Kna1Row = result?;

                // Handle missing SAP data by providing a default "Unknown" value
                let kunnr = record.kunnr.unwrap_or_else(|| "Unknown".to_string());

                // FILTER: Skip embedding this row if it's already in the database
                if existing_kunnrs.contains(&kunnr) {
                    continue;
                }

                // Add to internal HashSet so we also catch duplicates *inside* the CSV file itself
                existing_kunnrs.insert(kunnr.clone());

                let name = record.name1.unwrap_or_else(|| "Unknown".to_string());
                let city = record.ort01.unwrap_or_else(|| "Unknown".to_string());
                let country = record.land1.unwrap_or_else(|| "Unknown".to_string());

                // Serialize SAP row data into a natural language sentence for the embedding model
                let sentence = format!(
                    "Customer {} is named {} and is located in {}, {}.",
                    kunnr, name, city, country
                );

                // Retain the formatted sentence and the raw metadata
                documents.push(sentence.clone());
                records.push((name, city, kunnr));

                // Limit ingestion to the first 50 rows for the initial spike
                if documents.len() >= 50 {
                    break;
                }
            }

            if documents.is_empty() {
                println!("No new customers found. Exiting early to save CPU cycles.");
                return Ok(());
            }

            // ONLY load the massive Hugging Face model if we actually have new data to embed
            println!("Loading embedding model (BAAI/bge-base-en-v1.5) via pure Rust Candle...");
            let (model, tokenizer) = load_model()?;

            println!("Embedding {} rows...", documents.len());
    // Generate embeddings for the batch of sentences
    let embeddings = get_embeddings(&documents, &tokenizer, &model)?;

    println!("Saving vectors to LanceDB...");
    
    // Define the Apache Arrow Schema here to enforce strict data types
    let schema = Arc::new(Schema::new(vec![
        Field::new("kunnr", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("city", DataType::Utf8, false),
        Field::new("sentence", DataType::Utf8, false),
        Field::new("vector", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            768, // BAAI/bge-base-en-v1.5 produces 768-dimensional vectors
        ), false),
    ]));
    println!("LanceDB schema defined successfully.");

    // Convert Rust Vecs into strict Arrow Arrays
    let kunnr_array = StringArray::from(records.iter().map(|r| Some(r.2.clone())).collect::<Vec<_>>());
    let name_array = StringArray::from(records.iter().map(|r| Some(r.0.clone())).collect::<Vec<_>>());
    let city_array = StringArray::from(records.iter().map(|r| Some(r.1.clone())).collect::<Vec<_>>());
    let sentence_array = StringArray::from(documents);

    // Build the fixed size list array for the 768-dimensional vectors
    let mut vector_builder = FixedSizeListBuilder::new(PrimitiveBuilder::<Float32Type>::new(), 768);
    for emb in embeddings {
        vector_builder.values().append_slice(&emb);
        vector_builder.append(true);
    }
    let vector_array = vector_builder.finish();

    // Bundle them into a RecordBatch
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(kunnr_array),
            Arc::new(name_array),
            Arc::new(city_array),
            Arc::new(sentence_array),
            Arc::new(vector_array),
        ],
    ).map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    // Create the table or append if it already exists
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    match db.open_table("customers").execute().await {
        Ok(table) => {
            // ==========================================
            // 2. Database-Level Safety Net
            // ==========================================
            // Even though we filtered in memory, we use merge_insert (Upsert) here.
            // This prevents race conditions if two scripts run simultaneously,
            // ensuring absolute data integrity at the database hardware layer.
            println!("Table 'customers' already exists. Inserting new SAP records only...");
            let mut builder = table.merge_insert(&["kunnr"]);
            builder.when_not_matched_insert_all();
            builder.execute(Box::new(batches))
                .await
                .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        }
        Err(_) => {
            println!("Creating new 'customers' table...");
            db.create_table("customers", batches)
                .execute()
                .await
                .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        }
    }

    println!("Successfully ingested data into LanceDB 'customers' table!");

        }
        Commands::Ask { query } => {
            println!(">>> Executing ASK command with query: {}", query);
            
            // TODO: CODE_REVIEW - Review this query block next session
            // ==========================================
            // 1. Model Loading
            // ==========================================
            println!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
            let (model, tokenizer) = load_model()?;

            // ==========================================
            // 2. Query Embedding
            // ==========================================
            println!("Embedding search query...");
            let embeddings = get_embeddings(&[query.clone()], &tokenizer, &model)?;
            let query_vector = embeddings.into_iter().next().ok_or("Failed to generate embedding")?;

            // ==========================================
            // 3. LanceDB Connection
            // ==========================================
            println!("Connecting to LanceDB...");
            let db = lancedb::connect("data/sap_vectors").execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
            let table = db.open_table("customers").execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

            // ==========================================
            // 4. Vector Search Execution
            // ==========================================
            println!("Executing semantic search...");
            use futures::StreamExt; // Required to iterate over LanceDB's async stream
            let mut stream = table.query().nearest_to(query_vector).unwrap().limit(5).execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

            // ==========================================
            // 5. Result Presentation
            // ==========================================
            println!("\n--- Search Results ---");
            while let Some(result) = stream.next().await {
                let batch = result.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
                
                // Safely attempt to cast all columns, returning None if any fail
                let name_arr = batch.column_by_name("name").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
                let city_arr = batch.column_by_name("city").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
                let kunnr_arr = batch.column_by_name("kunnr").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
                let dist_arr = batch.column_by_name("_distance").and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());

                // Only print the results if ALL columns were safely found and casted
                if let (Some(names), Some(cities), Some(kunnrs), Some(distances)) = (name_arr, city_arr, kunnr_arr, dist_arr) {
                    for i in 0..batch.num_rows() {
                        let name = names.value(i);
                        let city = cities.value(i);
                        let kunnr = kunnrs.value(i);
                        let distance = distances.value(i);
                        
                        println!("Distance: {:.4} | [{}] {} ({})", distance, kunnr, name, city);
                    }
                } else {
                    // Log the error to STDERR but don't crash the application!
                    eprintln!("WARNING: Failed to read database columns. Search results skipped.");
                }
            }
        }
        Commands::AskSql { query } => {
            println!(">>> Executing ASK-SQL command with query: {}", query);
            
            execute_sql_query(&query).await?;
        }
        Commands::AskAiSql { query } => {
            println!(">>> Executing ASK-AISQL with question: '{}'", query);
            
            // 1. Build the constrained prompt
            let full_prompt = build_routing_prompt(&query);
            
            // 2. Ask the LLM to decide the route and generate SQL
            let raw_json_response = ask_llm(&full_prompt).await?;
            println!("LLM Raw Response: {}", raw_json_response);
            
            // 3. Parse the JSON safely
            let decision: RouterDecision = serde_json::from_str(&raw_json_response)?;
            
            // 4. Route the execution
            if decision.route == "SQL" {
                println!("Routing engine detected analytical intent. Executing DataFusion Zero-ETL...");
                execute_sql_query(&decision.query).await?;
            } else {
                println!("Routing engine detected semantic intent. Triggering LanceDB Vector Search...");
                // For right now, we just print the Semantic message. 
                // In a production app, we would call the Semantic search logic from Step 3 here.
            }
        }
    }

    Ok(())
}
