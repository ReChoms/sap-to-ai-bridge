use crate::models::{OllamaRequest, OllamaResponse};
use std::error::Error;

/// Sends a prompt to the local Ollama server running Llama 3.2
pub async fn ask_llm(prompt: &str) -> Result<String, Box<dyn Error>> {
    let req_body = OllamaRequest {
        model: "llama3.2:latest".to_string(),
        prompt: prompt.to_string(),
        stream: false,
    };

    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:11434/api/generate")
        .json(&req_body)
        .send()
        .await?
        .json::<OllamaResponse>()
        .await?;
    Ok(res.response)
}

pub fn build_routing_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and decide if it requires exact SQL or SEMANTIC search.\n\nDatabase Schema for table `kna1`:\n- kunnr (String): Customer Number / ID\n- name1 (String): Customer Name\n- ort01 (String): City\n- land1 (String): Country Code (e.g., 'US', 'DE')\n\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have two keys: \"route\" (either \"SQL\" or \"SEMANTIC\") and \"query\" (the generated SQL string, or blank).\n\nExamples:\nQ: \"How many customers are in Berlin?\"\nA: {{\"route\": \"SQL\", \"query\": \"SELECT count(*) FROM kna1 WHERE ort01 = 'Berlin'\"}}\n\nQ: \"Show me the names of 5 customers in the US.\"\nA: {{\"route\": \"SQL\", \"query\": \"SELECT name1 FROM kna1 WHERE land1 = 'US' LIMIT 5\"}}\n\nQ: \"Find customers who are large tech manufacturers.\"\nA: {{\"route\": \"SEMANTIC\", \"query\": \"\"}}\n\nUser Question: \"{}\"\nA: ",
        user_question
    )
}

pub fn build_sql_prompt(user_question: &str) -> String {
    format!(
        "You are an expert SAP data engineer. Read the user's question and write the exact SQL query required.\n\nDatabase Schema for table `kna1`:\n- kunnr (String): Customer Number / ID\n- name1 (String): Customer Name\n- ort01 (String): City\n- land1 (String): Country Code (e.g., 'US', 'DE')\n\nRULES:\n1. You must ONLY output raw JSON. Do not wrap it in markdown. Do not add conversational text.\n2. The JSON must have one key: \"query\" containing the generated SQL string.\n\nUser Question: \"{}\"\nA: ",
        user_question
    )
}
