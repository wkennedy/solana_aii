use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct InferenceData {
    input: String,
    output: String,
    model_version: String,
    metadata: HashMap<String, serde_json::Value>,
    timestamp: DateTime<Utc>,
    input_hash: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Block {
    index: u64,
    timestamp: DateTime<Utc>,
    inference_data: InferenceData,
    previous_hash: String,
    hash: String,
}

impl Block {
    fn new(index: u64, inference_data: InferenceData, previous_hash: String) -> Self {
        let timestamp = Utc::now();
        let mut block = Block {
            index,
            timestamp,
            inference_data,
            previous_hash,
            hash: String::new(),
        };
        block.hash = block.calculate_hash();
        block
    }

    fn calculate_hash(&self) -> String {
        let block_string = serde_json::json!({
            "index": self.index,
            "timestamp": self.timestamp,
            "inference_data": self.inference_data,
            "previous_hash": self.previous_hash,
        });

        let mut hasher = Sha256::new();
        hasher.update(block_string.to_string().as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

#[derive(Debug)]
struct AIBlockchain {
    chain: Vec<Block>,
}

impl AIBlockchain {
    fn new() -> Self {
        let genesis_data = InferenceData {
            input: String::from("genesis"),
            output: String::from("genesis"),
            model_version: String::from("0.0.0"),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            input_hash: String::from("0"),
        };

        let genesis_block = Block::new(0, genesis_data, String::from("0"));

        AIBlockchain {
            chain: vec![genesis_block],
        }
    }

    fn get_latest_block(&self) -> &Block {
        self.chain.last().unwrap()
    }

    fn add_inference(
        &mut self,
        input: String,
        output: String,
        model_version: String,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<&Block, String> {
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        let input_hash = format!("{:x}", hasher.finalize());

        let inference_data = InferenceData {
            input,
            output,
            model_version,
            metadata,
            timestamp: Utc::now(),
            input_hash,
        };

        let block = Block::new(
            self.chain.len() as u64,
            inference_data,
            self.get_latest_block().hash.clone(),
        );

        self.chain.push(block);
        Ok(self.get_latest_block())
    }

    fn is_chain_valid(&self) -> bool {
        for i in 1..self.chain.len() {
            let current = &self.chain[i];
            let previous = &self.chain[i - 1];

            if current.hash != current.calculate_hash() {
                return false;
            }

            if current.previous_hash != previous.hash {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Serialize)]
struct InferenceResult {
    block_hash: String,
    timestamp: DateTime<Utc>,
    inference_data: InferenceData,
}

fn log_inference(
    blockchain: &mut AIBlockchain,
    input: String,
    output: String,
    model_version: String,
    metadata: HashMap<String, serde_json::Value>,
) -> Result<InferenceResult, String> {
    let block = blockchain.add_inference(input, output, model_version, metadata)?;

    Ok(InferenceResult {
        block_hash: block.hash.clone(),
        timestamp: block.timestamp,
        inference_data: block.inference_data.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference() {
        // Initialize blockchain
        let mut ai_blockchain = AIBlockchain::new();

        // Create sample metadata
        let mut metadata = HashMap::new();
        metadata.insert("confidence".to_string(), serde_json::json!(0.98));
        metadata.insert("latency_ms".to_string(), serde_json::json!(150));
        metadata.insert("temperature".to_string(), serde_json::json!(0.7));

        // Log a sample inference
        match log_inference(
            &mut ai_blockchain,
            "What is the capital of France?".to_string(),
            "Paris".to_string(),
            "gpt-4".to_string(),
            metadata,
        ) {
            Ok(result) => {
                println!("Chain valid: {}", ai_blockchain.is_chain_valid());
                println!("Latest block hash: {}", result.block_hash);
            }
            Err(e) => println!("Error logging inference: {}", e),
        }
    }
}
