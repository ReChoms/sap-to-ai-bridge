#[cfg(test)]
mod integration_tests {
    use crate::db::execute_semantic_search;
    use crate::ingest::execute_ingestion;
    use crate::llm::{ask_llm, build_routing_prompt};
    use crate::models::RouterDecision;
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;

    #[tokio::test]
    async fn test_dummy_data_pipeline() {
        // 1. Create a dummy CSV
        let csv_path = "/tmp/dummy_kna1.csv";
        let mut file = File::create(csv_path).unwrap();
        writeln!(file, "kunnr,name1,ort01,land1").unwrap();
        writeln!(file, "DUMMY001,Acme Corp,Berlin,DE").unwrap();
        writeln!(file, "DUMMY002,Global Tech,Munich,DE").unwrap();
        
        // 2. Trigger Ingestion WITHOUT overwrite (safely appending to your real database)
        let result = execute_ingestion(csv_path, false, 2).await;
        assert!(result.is_ok(), "Ingestion pipeline panicked");

        // 3. Verify LanceDB files exist on disk
        let db_path = Path::new("data/sap_vectors/customers.lance");
        assert!(db_path.exists(), "LanceDB failed to write physical files");
        
        // Cleanup
        std::fs::remove_file(csv_path).unwrap();
    }

    #[tokio::test]
    async fn test_real_sap_pipeline() {
        // 1. Ingest the real data WITHOUT overwrite. 
        // Because of our deduplication loop, this finishes instantly if the 20,210 rows already exist!
        let csv_path = "data/kna1.csv";
        let result = execute_ingestion(csv_path, false, 128).await;
        assert!(result.is_ok(), "Ingestion pipeline panicked");

        // 2. Query the real data
        let result = execute_semantic_search("technology companies in Berlin").await;
        assert!(result.is_ok(), "Semantic search execution failed");
    }

    #[tokio::test]
    #[ignore = "Requires Ollama daemon running on localhost:11434"]
    async fn test_llm_router_determinism() {
        // Test 1: Exact SQL match
        let sql_prompt = build_routing_prompt("How many customers are in Berlin?");
        let sql_json = ask_llm(&sql_prompt).await.unwrap();
        let sql_decision: RouterDecision = serde_json::from_str(&sql_json).unwrap();
        assert_eq!(sql_decision.route, "SQL", "LLM Hallucinated on SQL query");

        // Test 2: Fuzzy Semantic match
        let fuzzy_prompt = build_routing_prompt("Find companies that bake bread.");
        let fuzzy_json = ask_llm(&fuzzy_prompt).await.unwrap();
        let fuzzy_decision: RouterDecision = serde_json::from_str(&fuzzy_json).unwrap();
        assert_eq!(fuzzy_decision.route, "SEMANTIC", "LLM Hallucinated on Semantic query");
    }
}
