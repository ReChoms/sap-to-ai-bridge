use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;
use clap::{Parser, Subcommand};
use std::sync::Arc;
use arrow_schema::{Field, Schema, DataType};
use arrow_array::{RecordBatch, StringArray, FixedSizeListArray, RecordBatchIterator};
use arrow_array::builder::{PrimitiveBuilder, FixedSizeListBuilder};
use arrow_array::types::Float32Type;
use lancedb::query::{ExecutableQuery, QueryBase};

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse CLI arguments
    let cli = Cli::parse();

    match &cli.command {
        Commands::Ingest { file, overwrite } => {
            println!(">>> Executing INGEST command on file: {}", file);
            let csv_path = file.clone();

    println!("Loading embedding model (BAAI/bge-base-en-v1.5) via pure Rust Candle...");
    let (model, tokenizer) = load_model()?;

    println!(
        "DEBUG: I am trying to open the file exactly at: {:?}",
        csv_path
    );

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
        println!("No data found.");
        return Ok(());
    }

    println!("Embedding {} rows...", documents.len());
    // Generate embeddings for the batch of sentences
    let embeddings = get_embeddings(&documents, &tokenizer, &model)?;

    println!("Connecting to LanceDB and saving vectors...");
    
    // Connect to local LanceDB instance in the data folder
    let db = lancedb::connect("data/sap_vectors").execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
    
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

    if *overwrite {
        println!("Overwrite flag detected. Dropping existing 'customers' table...");
        let _ = db.drop_table("customers").await;
    }

    // Create the table or append if it already exists
    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    match db.open_table("customers").execute().await {
        Ok(table) => {
            println!("Table 'customers' already exists. Appending new data...");
            table.add(batches).execute().await.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
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
                
                let name_array = batch.column_by_name("name").unwrap().as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
                let city_array = batch.column_by_name("city").unwrap().as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
                let kunnr_array = batch.column_by_name("kunnr").unwrap().as_any().downcast_ref::<arrow_array::StringArray>().unwrap();
                let distance_array = batch.column_by_name("_distance").unwrap().as_any().downcast_ref::<arrow_array::Float32Array>().unwrap();

                for i in 0..batch.num_rows() {
                    let name = name_array.value(i);
                    let city = city_array.value(i);
                    let kunnr = kunnr_array.value(i);
                    let distance = distance_array.value(i);
                    
                    println!("Distance: {:.4} | [{}] {} ({})", distance, kunnr, name, city);
                }
            }
        }
    }

    Ok(())
}
