use arrow_array::builder::{FixedSizeListBuilder, PrimitiveBuilder};
use arrow_array::types::Float32Type;
use arrow_array::{Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::Schema;
use datafusion::prelude::*;
use futures::StreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::error::Error;
use std::sync::Arc;
use tracing::{info, warn};

use crate::embeddings::{get_embeddings, load_model};

pub async fn insert_batch(
    db: &lancedb::Connection,
    schema: Arc<Schema>,
    records: &[(String, String, String)],
    documents: &[String],
    embeddings: Vec<Vec<f32>>,
) -> Result<(), Box<dyn Error>> {
    let kunnr_array = StringArray::from(
        records
            .iter()
            .map(|r| Some(r.2.clone()))
            .collect::<Vec<_>>(),
    );
    let name_array = StringArray::from(
        records
            .iter()
            .map(|r| Some(r.0.clone()))
            .collect::<Vec<_>>(),
    );
    let city_array = StringArray::from(
        records
            .iter()
            .map(|r| Some(r.1.clone()))
            .collect::<Vec<_>>(),
    );
    let sentence_array = StringArray::from(documents.to_vec());

    let mut vector_builder =
        FixedSizeListBuilder::new(PrimitiveBuilder::<Float32Type>::new(), 768);
    for emb in embeddings {
        vector_builder.values().append_slice(&emb);
        vector_builder.append(true);
    }
    let vector_array = vector_builder.finish();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(kunnr_array),
            Arc::new(name_array),
            Arc::new(city_array),
            Arc::new(sentence_array),
            Arc::new(vector_array),
        ],
    )
    .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    let batches = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    match db.open_table("customers").execute().await {
        Ok(table) => {
            let mut builder = table.merge_insert(&["kunnr"]);
            builder.when_not_matched_insert_all();
            builder
                .execute(Box::new(batches))
                .await
                .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        }
        Err(_) => {
            db.create_table("customers", batches)
                .execute()
                .await
                .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        }
    }
    Ok(())
}

pub async fn execute_sql_query(query: &str) -> Result<(), Box<dyn Error>> {
    info!("Spinning up Apache DataFusion Engine (Zero-Copy)...");
    let ctx = SessionContext::new();

    info!("Registering data/kna1.csv as virtual SQL table...");
    ctx.register_csv("kna1", "data/kna1.csv", CsvReadOptions::new())
        .await?;

    info!("Executing SQL: {}", query);
    let df = ctx.sql(query).await?;
    df.show().await?;

    Ok(())
}

pub async fn execute_semantic_search(query: &str) -> Result<(), Box<dyn Error>> {
    info!("Loading embedding model (BAAI/bge-base-en-v1.5)...");
    let (model, tokenizer) = load_model()?;

    info!("Embedding search query...");
    let embeddings = get_embeddings(&[query.to_string()], &tokenizer, &model)?;
    let query_vector = embeddings
        .into_iter()
        .next()
        .ok_or("Failed to generate embedding")?;

    info!("Connecting to LanceDB...");
    let db = lancedb::connect("data/sap_vectors")
        .execute()
        .await
        .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
    let table = db
        .open_table("customers")
        .execute()
        .await
        .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    info!("Executing semantic search...");
    let mut stream = table
        .query()
        .nearest_to(query_vector)
        .unwrap()
        .limit(5)
        .execute()
        .await
        .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

    info!("--- Search Results ---");

    #[derive(serde::Serialize)]
    struct SearchResultPayload {
        distance: f32,
        kunnr: String,
        name: String,
        city: String,
    }

    while let Some(result) = stream.next().await {
        let batch = result.map_err(|e| Box::<dyn Error>::from(e.to_string()))?;

        let name_arr = batch
            .column_by_name("name")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let city_arr = batch
            .column_by_name("city")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let kunnr_arr = batch
            .column_by_name("kunnr")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let dist_arr = batch
            .column_by_name("_distance")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::Float32Array>());

        if let (Some(names), Some(cities), Some(kunnrs), Some(distances)) =
            (name_arr, city_arr, kunnr_arr, dist_arr)
        {
            for i in 0..batch.num_rows() {
                let payload = SearchResultPayload {
                    distance: distances.value(i),
                    kunnr: kunnrs.value(i).to_string(),
                    name: names.value(i).to_string(),
                    city: cities.value(i).to_string(),
                };
                println!("{}", serde_json::to_string(&payload)?);
            }
        } else {
            warn!("WARNING: Failed to read database columns. Search results skipped.");
        }
    }

    Ok(())
}
