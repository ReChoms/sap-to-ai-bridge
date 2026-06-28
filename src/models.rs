use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Kna1Row {
    pub kunnr: Option<String>,
    pub name1: Option<String>,
    pub ort01: Option<String>,
    pub land1: Option<String>,
}

#[derive(Serialize)]
pub struct OllamaRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
}

#[derive(Deserialize)]
pub struct OllamaResponse {
    pub response: String,
}

#[derive(Deserialize, Debug)]
pub struct RouterDecision {
    pub route: String,
    pub query: String,
}

#[derive(Deserialize, Debug)]
pub struct SqlResponse {
    pub query: String,
}
