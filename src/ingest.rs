use arrow_array::Array;
use arrow_schema::{DataType, Field, Schema};
use lancedb::query::ExecutableQuery;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;
use tracing::info;

use crate::db::insert_batch;
use crate::embeddings::{get_embeddings, load_model};
use crate::models::Kna1Row;

/// Main entry point for the ingestion pipeline
pub async fn execute_ingestion(
    csv_path: &str,
    overwrite: bool,
    batch_size: usize,
) -> Result<(), Box<dyn Error>> {
    info!(">>> Executing INGEST command on file: {}", csv_path);

    info!("Connecting to LanceDB...");
    let db = lancedb::connect("data/sap_vectors")
        .execute()
        .await
        .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    if overwrite {
        info!("Overwrite flag detected. Dropping existing 'customers' table...");
        let _ = db.drop_table("customers").await;
    }

    let existing_kunnrs = fetch_existing_kunnrs(&db).await;

    info!("Loading embedding model (BAAI/bge-base-en-v1.5) via pure Rust Candle...");
    let (model, tokenizer) = load_model()?;

    let schema = define_schema();

    info!("Reading {} with dynamic chunk size {}...", csv_path, batch_size);
    let file_stream = File::open(csv_path)
        .map_err(|e| format!("File load went wrong. Rust shows the following error: {}", e))?;
    let mut rdr = csv::Reader::from_reader(file_stream);

    process_csv_in_batches(&mut rdr, &db, schema, &model, &tokenizer, existing_kunnrs, batch_size).await?;

    Ok(())
}

/// Helper function to pull existing database keys and prevent duplicate CPU math
async fn fetch_existing_kunnrs(db: &lancedb::Connection) -> HashSet<String> {
    let mut existing_kunnrs = HashSet::new();
    if let Ok(table) = db.open_table("customers").execute().await {
        info!("Table exists. Scanning existing records to prevent redundant CPU embeddings...");
        use futures::StreamExt;
        if let Ok(mut stream) = table.query().execute().await {
            while let Some(batch) = stream.next().await {
                if let Ok(batch) = batch {
                    if let Some(col) = batch.column_by_name("kunnr") {
                        if let Some(kunnr_array) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                            for i in 0..kunnr_array.len() {
                                existing_kunnrs.insert(kunnr_array.value(i).to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    existing_kunnrs
}

/// Helper function defining the strict Apache Arrow Memory Schema
fn define_schema() -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("kunnr", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("city", DataType::Utf8, false),
        Field::new("sentence", DataType::Utf8, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                768,
            ),
            false,
        ),
    ]));
    info!("LanceDB schema defined successfully.");
    schema
}

/// The core loop that processes CSV records, buffers them, and pushes batches to LanceDB
async fn process_csv_in_batches(
    rdr: &mut csv::Reader<File>,
    db: &lancedb::Connection,
    schema: Arc<Schema>,
    model: &candle_transformers::models::bert::BertModel,
    tokenizer: &tokenizers::Tokenizer,
    mut existing_kunnrs: HashSet<String>,
    batch_size: usize,
) -> Result<(), Box<dyn Error>> {
    let mut documents = Vec::new();
    let mut records = Vec::new();
    let mut total_inserted = 0;

    for result in rdr.deserialize() {
        let record: Kna1Row = result?;
        let kunnr = record.kunnr.unwrap_or_else(|| "Unknown".to_string());

        if existing_kunnrs.contains(&kunnr) {
            continue;
        }
        existing_kunnrs.insert(kunnr.clone());

        let name = record.name1.unwrap_or_else(|| "Unknown".to_string());
        let city = record.ort01.unwrap_or_else(|| "Unknown".to_string());
        let country = record.land1.unwrap_or_else(|| "Unknown".to_string());

        documents.push(format!("Customer {} is named {} and is located in {}, {}.", kunnr, name, city, country));
        records.push((name, city, kunnr));

        if documents.len() >= batch_size {
            info!("Batch full ({} rows). Generating embeddings...", documents.len());
            let embeddings = get_embeddings(&documents, tokenizer, model)?;
            insert_batch(db, schema.clone(), &records, &documents, embeddings).await?;
            total_inserted += documents.len();

            documents.clear();
            records.clear();
        }
    }

    if !documents.is_empty() {
        info!("Inserting remainder batch ({} rows)...", documents.len());
        let embeddings = get_embeddings(&documents, tokenizer, model)?;
        insert_batch(db, schema.clone(), &records, &documents, embeddings).await?;
        total_inserted += documents.len();
    }

    info!("Successfully ingested {} total new records into LanceDB 'customers' table!", total_inserted);
    Ok(())
}
