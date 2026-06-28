mod cli;
mod db;
mod debug;
mod embeddings;
mod ingest;
mod llm;
mod models;

#[cfg(test)]
mod tests;

use clap::Parser;
use std::error::Error;
use tracing::info;

use crate::cli::{Cli, Commands};
use crate::db::{execute_semantic_search, execute_sql_query};
use crate::ingest::execute_ingestion;
use crate::llm::{ask_llm, build_routing_prompt, build_sql_prompt};
use crate::models::{RouterDecision, SqlResponse};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    debug::init_logger();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Ingest { file, overwrite, batch_size } => {
            execute_ingestion(file, *overwrite, *batch_size).await?;
        }
        Commands::AskSemantic { query } => {
            info!(">>> Executing ASK-SEMANTIC command");
            execute_semantic_search(&query).await?;
        }
        Commands::Ask { query } => {
            info!(">>> Executing ASK (ROUTER) command");
            let full_prompt = build_routing_prompt(&query);
            let raw_json = ask_llm(&full_prompt).await?;
            let decision: RouterDecision = serde_json::from_str(&raw_json)?;

            if decision.route == "SQL" {
                execute_sql_query(&decision.query).await?;
            } else {
                execute_semantic_search(&query).await?;
            }
        }
        Commands::AskSql { query } => {
            info!(">>> Executing ASK-SQL command");
            execute_sql_query(&query).await?;
        }
        Commands::AskAiSql { query } => {
            info!(">>> Executing ASK-AISQL command");
            let full_prompt = build_sql_prompt(&query);

            let raw_json = ask_llm(&full_prompt).await?;
            let response: SqlResponse = serde_json::from_str(&raw_json)
                .expect("Failed to parse JSON from LLM. LLM generated invalid output.");

            execute_sql_query(&response.query).await?;
        }
    }

    Ok(())
}
