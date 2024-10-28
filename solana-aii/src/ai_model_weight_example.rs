use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelWeight {
    layer_name: String,
    weight_hash: String,
    shape: Vec<usize>,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelArchitecture {
    name: String,
    version: String,
    weights: Vec<ModelWeight>,
    architecture_hash: String,
}

impl ModelArchitecture {
    fn calculate_architecture_hash(&self) -> String {
        let mut hasher = Sha256::new();
        // Hash the concatenated weight hashes in order
        for weight in &self.weights {
            hasher.update(weight.weight_hash.as_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    fn verify_integrity(&self) -> bool {
        self.architecture_hash == self.calculate_architecture_hash()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelVerification {
    model_architecture: ModelArchitecture,
    verification_timestamp: DateTime<Utc>,
    verified: bool,
    verification_metadata: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct InferenceData {
    input: String,
    output: String,
    model_version: String,
    metadata: HashMap<String, serde_json::Value>,
    timestamp: DateTime<Utc>,
    input_hash: String,
    model_verification: ModelVerification,
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
struct WeightVerifier {
    trusted_model_hashes: HashMap<String, String>,
}

impl WeightVerifier {
    fn new() -> Self {
        WeightVerifier {
            trusted_model_hashes: HashMap::new(),
        }
    }

    fn add_trusted_model(&mut self, model_name: String, architecture_hash: String) {
        self.trusted_model_hashes
            .insert(model_name, architecture_hash);
    }

    fn verify_weights(&self, model: &ModelArchitecture) -> bool {
        // Check if model is in trusted list
        if let Some(trusted_hash) = self.trusted_model_hashes.get(&model.name) {
            // Verify the architecture hash matches trusted hash
            if trusted_hash != &model.architecture_hash {
                return false;
            }

            // Verify individual weights
            model.verify_integrity()
        } else {
            false
        }
    }
}

#[derive(Debug)]
struct AIBlockchain {
    chain: Vec<Block>,
    weight_verifier: WeightVerifier,
}

impl AIBlockchain {
    fn new() -> Self {
        let mut weight_verifier = WeightVerifier::new();
        // Add some trusted model hashes (in practice, these would come from a trusted source)
        weight_verifier.add_trusted_model(
            "gpt-4".to_string(),
            "0b80beee67bb324b42487cbebedf6918afe2626d2e40efe0a31ec5eba447bbbd".to_string(),
        );

        let genesis_data = InferenceData {
            input: String::from("genesis"),
            output: String::from("genesis"),
            model_version: String::from("0.0.0"),
            metadata: HashMap::new(),
            timestamp: Utc::now(),
            input_hash: String::from("0"),
            model_verification: ModelVerification {
                model_architecture: ModelArchitecture {
                    name: String::from("genesis"),
                    version: String::from("0.0.0"),
                    weights: vec![],
                    architecture_hash: String::from("0"),
                },
                verification_timestamp: Utc::now(),
                verified: true,
                verification_metadata: HashMap::new(),
            },
        };

        let genesis_block = Block::new(0, genesis_data, String::from("0"));

        AIBlockchain {
            chain: vec![genesis_block],
            weight_verifier,
        }
    }

    fn verify_model_weights(&self, model: &ModelArchitecture) -> ModelVerification {
        let verified = self.weight_verifier.verify_weights(model);
        let mut verification_metadata = HashMap::new();

        verification_metadata.insert(
            "verification_method".to_string(),
            "hash_verification".to_string(),
        );

        ModelVerification {
            model_architecture: model.clone(),
            verification_timestamp: Utc::now(),
            verified,
            verification_metadata,
        }
    }

    fn add_inference(
        &mut self,
        input: String,
        output: String,
        model: ModelArchitecture,
        metadata: HashMap<String, serde_json::Value>,
    ) -> Result<&Block, String> {
        // Verify model weights before adding inference
        let model_verification = self.verify_model_weights(&model);

        if !model_verification.verified {
            return Err("Model verification failed".to_string());
        }

        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        let input_hash = format!("{:x}", hasher.finalize());

        let inference_data = InferenceData {
            input,
            output,
            model_version: model.version.clone(),
            metadata,
            timestamp: Utc::now(),
            input_hash,
            model_verification,
        };

        let block = Block::new(
            self.chain.len() as u64,
            inference_data,
            self.get_latest_block().hash.clone(),
        );

        self.chain.push(block);
        Ok(self.get_latest_block())
    }

    fn get_latest_block(&self) -> &Block {
        self.chain.last().unwrap()
    }
}

// Example of creating model weights for verification
fn create_sample_model_architecture() -> ModelArchitecture {
    let mut weights = Vec::new();

    // Sample weights for different layers
    let layers = vec![
        ("embedding", vec![50000, 768]),
        ("attention_1", vec![768, 768]),
        ("ffn_1", vec![768, 3072]),
    ];

    for (layer_name, shape) in layers {
        let mut hasher = Sha256::new();
        // In practice, you would hash the actual weight tensors
        hasher.update(format!("{:?}", shape).as_bytes());

        let mut metadata = HashMap::new();
        metadata.insert("initialization".to_string(), "normal".to_string());

        weights.push(ModelWeight {
            layer_name: layer_name.to_string(),
            weight_hash: format!("{:x}", hasher.finalize()),
            shape: shape.to_vec(),
            metadata,
        });
    }

    let mut model = ModelArchitecture {
        name: "gpt-4".to_string(),
        version: "1.0.0".to_string(),
        weights,
        architecture_hash: String::new(),
    };

    model.architecture_hash = model.calculate_architecture_hash();
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_weight() {
        let mut ai_blockchain = AIBlockchain::new();
        let model = create_sample_model_architecture();
        println!("Model Hash: {:?}", model.architecture_hash);

        // Create sample metadata
        let mut metadata = HashMap::new();
        metadata.insert("confidence".to_string(), serde_json::json!(0.98));

        match ai_blockchain.add_inference(
            "What is the capital of France?".to_string(),
            "Paris".to_string(),
            model,
            metadata,
        ) {
            Ok(block) => {
                println!("Inference added successfully!");
                println!(
                    "Model verified: {}",
                    block.inference_data.model_verification.verified
                );
                println!("Block hash: {}", block.hash);
            }
            Err(e) => println!("Error adding inference: {}", e),
        }
    }
}
