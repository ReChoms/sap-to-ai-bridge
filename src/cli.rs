use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "sap-to-ai-bridge")]
#[command(about = "Bridging SAP ERP data with Semantic AI Search", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Ingest a CSV file from SAP into the Vector Database
    Ingest {
        /// The path to the CSV file (e.g. data/kna1.csv)
        #[arg(short, long)]
        file: String,
        /// Overwrite the existing database instead of appending
        #[arg(short, long)]
        overwrite: bool,
        /// The dynamic batch size for ingestion chunking (default: 256)
        #[arg(short, long, default_value_t = 256)]
        batch_size: usize,
    },
    /// The Primary Router: dynamically chooses Semantic or SQL
    Ask {
        query: String,
    },
    /// Force a Semantic Vector Search
    AskSemantic {
        query: String,
    },
    /// Execute a raw SQL query against the SAP data
    AskSql {
        query: String,
    },
    /// Force the LLM to generate and run a SQL query
    AskAiSql {
        query: String,
    },
}
