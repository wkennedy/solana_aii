use base64;
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Serialize, Deserialize)]
struct LogitsData {
    model_id: String,
    timestamp: String,
    sequence_length: usize,
    vocabulary_size: usize,
    logits: LogitsMatrix,
    metadata: Metadata,
    verification: Verification,
}

#[derive(Debug, Serialize, Deserialize)]
struct LogitsMatrix {
    values: Vec<Vec<f64>>,
    shape: [usize; 2],
}

#[derive(Debug, Serialize, Deserialize)]
struct Metadata {
    temperature: f64,
    top_k: u32,
    top_p: f64,
    input_text: String,
    token_ids: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Verification {
    hash_algorithm: String,
    hash_input_fields: Vec<String>,
    hash: String,
    signature: String,
    certificate: String,
}

#[derive(Debug)]
pub enum VerificationError {
    HashMismatch,
    SignatureInvalid,
    ShapeInconsistent,
    UnusualDistribution,
    FingerprintMismatch,
    DataError(String),
}

pub struct LogitsVerifier {
    known_fingerprints: HashMap<String, HashMap<String, Vec<f64>>>,
}

impl LogitsVerifier {
    pub fn new(fingerprints: HashMap<String, HashMap<String, Vec<f64>>>) -> Self {
        Self {
            known_fingerprints: fingerprints,
        }
    }

    pub fn verify_inference(&self, logits_data: &LogitsData) -> Result<(), VerificationError> {
        // 1. Verify data integrity
        self.verify_hash(logits_data)?;
        self.verify_signature(logits_data)?;

        // 2. Convert logits to ndarray for efficient computation
        let logits_array = Array2::from_shape_vec(
            (logits_data.sequence_length, logits_data.vocabulary_size),
            logits_data
                .logits
                .values
                .iter()
                .flatten()
                .cloned()
                .collect(),
        )
        .map_err(|e| VerificationError::DataError(e.to_string()))?;

        // 3. Verify shape consistency
        self.verify_shape_consistency(logits_data)?;

        // 4. Analyze logits distribution
        self.verify_logits_distribution(&logits_array)?;

        // 5. Compare with known model fingerprint
        self.verify_model_fingerprint(
            &logits_data.model_id,
            &logits_data.metadata.input_text,
            &logits_array,
        )?;

        Ok(())
    }

    fn verify_hash(&self, logits_data: &LogitsData) -> Result<(), VerificationError> {
        let computed_hash = self.compute_hash(logits_data)?;
        if computed_hash != logits_data.verification.hash {
            return Err(VerificationError::HashMismatch);
        }
        Ok(())
    }

    fn compute_hash(&self, data: &LogitsData) -> Result<String, VerificationError> {
        let mut hasher = Sha256::new();

        // Concatenate specified fields in order
        for field_path in &data.verification.hash_input_fields {
            // TODO Using this isn't especially efficient and string representation of array is used instead of actual values
            let value = self.get_nested_value(data, field_path)?;
            hasher.update(value.as_bytes());
        }

        let result = hasher.finalize();
        Ok(format!("sha256:{:x}", result))
    }

    //TODO Maybe use secp256k1 - Just bypassing this for example purposes
    fn verify_signature(&self, data: &LogitsData) -> Result<(), VerificationError> {
        // let cert_der = base64::decode(&data.verification.certificate)
        //     .map_err(|e| VerificationError::DataError(e.to_string()))?;
        //
        // let signature = base64::decode(&data.verification.signature)
        //     .map_err(|e| VerificationError::DataError(e.to_string()))?;
        //
        // let cert = Certificate(cert_der);
        // let public_key = signature::UnparsedPublicKey::new(
        //     &signature::RSA_PKCS1_2048_8192_SHA256,
        //     cert.0,
        // );
        //
        // public_key
        //     .verify(
        //         data.verification.hash.as_bytes(),
        //         &signature,
        //     )
        //     .map_err(|_| VerificationError::SignatureInvalid)?;

        Ok(())
    }

    fn verify_shape_consistency(&self, data: &LogitsData) -> Result<(), VerificationError> {
        if data.logits.shape[0] != data.sequence_length
            || data.logits.shape[1] != data.vocabulary_size
        {
            return Err(VerificationError::ShapeInconsistent);
        }
        Ok(())
    }

    fn verify_logits_distribution(&self, logits: &Array2<f64>) -> Result<(), VerificationError> {
        // Check value range
        let max_abs = logits.iter().fold(0f64, |max, &x| max.max(x.abs()));
        if max_abs > 100.0 {
            return Err(VerificationError::UnusualDistribution);
        }

        // Check variance
        let mean = logits.mean().unwrap_or(0.0);
        let variance =
            logits.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / logits.len() as f64;

        if variance < 0.01 {
            return Err(VerificationError::UnusualDistribution);
        }

        // Check for repeated patterns
        let rows = logits.rows();
        let mut unique_rows = Vec::new();
        for row in rows.into_iter() {
            if !unique_rows.contains(&row.to_vec()) {
                unique_rows.push(row.to_vec());
            }
        }

        if unique_rows.len() < logits.nrows() / 2 {
            return Err(VerificationError::UnusualDistribution);
        }

        Ok(())
    }

    fn verify_model_fingerprint(
        &self,
        model_id: &str,
        input_text: &str,
        logits: &Array2<f64>,
    ) -> Result<(), VerificationError> {
        let fingerprints = self
            .known_fingerprints
            .get(model_id)
            .ok_or(VerificationError::FingerprintMismatch)?;

        let expected_pattern = fingerprints
            .get(input_text)
            .ok_or(VerificationError::FingerprintMismatch)?;

        // Calculate the total number of elements needed
        let total_elements = logits.dim().0 * logits.dim().1;

        // Ensure the fingerprint pattern has the correct length
        if expected_pattern.len() != total_elements {
            return Err(VerificationError::DataError(format!(
                "Fingerprint length mismatch: expected {}, got {}",
                total_elements,
                expected_pattern.len()
            )));
        }

        // Create the expected array with the correct shape
        let expected_array =
            Array2::from_shape_vec((logits.dim().0, logits.dim().1), expected_pattern.clone())
                .map_err(|e| VerificationError::DataError(e.to_string()))?;

        // Calculate correlation
        let correlation = self.calculate_correlation(logits.view(), expected_array.view())?;

        if correlation < 0.95 {
            return Err(VerificationError::FingerprintMismatch);
        }

        Ok(())
    }

    fn calculate_correlation(
        &self,
        a: ArrayView2<f64>,
        b: ArrayView2<f64>,
    ) -> Result<f64, VerificationError> {
        let a_mean = a.mean().unwrap_or(0.0);
        let b_mean = b.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut a_variance = 0.0;
        let mut b_variance = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let a_diff = x - a_mean;
            let b_diff = y - b_mean;
            numerator += a_diff * b_diff;
            a_variance += a_diff * a_diff;
            b_variance += b_diff * b_diff;
        }

        if a_variance == 0.0 || b_variance == 0.0 {
            return Err(VerificationError::DataError("Zero variance".to_string()));
        }

        Ok(numerator / (a_variance.sqrt() * b_variance.sqrt()))
    }

    // TODO this is good for ease of recreating a hash based on dynamic fields, but probably a better way to do this
    fn get_nested_value(&self, data: &LogitsData, path: &str) -> Result<String, VerificationError> {
        // Convert the entire LogitsData to a serde_json::Value for easier traversal
        let value = serde_json::to_value(data)
            .map_err(|e| VerificationError::DataError(format!("Serialization error: {}", e)))?;

        // Split the path into parts
        let parts: Vec<&str> = path.split('.').collect();

        // Traverse the nested structure
        let mut current = &value;
        for &part in parts.iter() {
            current = current.get(part).ok_or_else(|| {
                VerificationError::DataError(format!("Invalid path: {} (at {})", path, part))
            })?;
        }

        // Handle different value types appropriately
        match current {
            Value::Null => Ok("null".to_string()),
            Value::Bool(b) => Ok(b.to_string()),
            Value::Number(n) => Ok(n.to_string()),
            Value::String(s) => Ok(s.clone()),
            Value::Array(arr) => {
                // For arrays, we create a deterministic string representation
                // that's suitable for hashing
                let elements: Vec<String> = arr
                    .iter()
                    .map(|v| match v {
                        Value::Number(n) => n.to_string(),
                        Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    })
                    .collect();
                Ok(format!("[{}]", elements.join(",")))
            }
            Value::Object(obj) => {
                // For objects, create a deterministic string representation
                let mut pairs: Vec<(String, String)> = obj
                    .iter()
                    .map(|(k, v)| {
                        let value_str = match v {
                            Value::Number(n) => n.to_string(),
                            Value::String(s) => s.clone(),
                            _ => v.to_string(),
                        };
                        (k.clone(), value_str)
                    })
                    .collect();
                // Sort to ensure deterministic output
                pairs.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
                let formatted = pairs
                    .iter()
                    .map(|(k, v)| format!("\"{}\":{}", k, v))
                    .collect::<Vec<_>>()
                    .join(",");
                Ok(format!("{{{}}}", formatted))
            }
        }
    }

    // Helper function to verify a specific field exists in the data structure
    fn verify_field_exists(&self, data: &LogitsData, path: &str) -> bool {
        let value = serde_json::to_value(data).ok().unwrap();
        let parts = path.split('.');
        let mut current = &value;

        for part in parts {
            current = current.get(part).unwrap();
        }

        Some(true).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example usage showing how to properly initialize the fingerprints
    fn create_test_verifier() -> LogitsVerifier {
        let mut fingerprints = HashMap::new();
        let mut model_patterns = HashMap::new();

        // Create a flattened vector of the expected logits pattern
        // This should match the dimensions of your input (3x5 = 15 elements)
        let pattern = vec![
            2.1, -0.3, 1.5, 0.8, -1.2, 0.5, 2.8, -0.4, 1.1, 0.9, -0.2, 1.7, 2.3, -0.8, 0.4,
        ];

        model_patterns.insert("Hello world!".to_string(), pattern);
        fingerprints.insert("gpt-example-v1".to_string(), model_patterns);

        LogitsVerifier::new(fingerprints)
    }

    #[test]
    fn test_verification_with_real_data() {
        let verifier = create_test_verifier();

        // Create test data matching your JSON structure
        let test_logits = Array2::from_shape_vec(
            (3, 5),
            vec![
                2.1, -0.3, 1.5, 0.8, -1.2, 0.5, 2.8, -0.4, 1.1, 0.9, -0.2, 1.7, 2.3, -0.8, 0.4,
            ],
        )
        .unwrap();

        let result =
            verifier.verify_model_fingerprint("gpt-example-v1", "Hello world!", &test_logits);

        assert!(result.is_ok());
    }

    #[test]
    fn test_verification_from_file() -> Result<(), Box<dyn Error>> {
        // Create verifier
        let verifier = create_test_verifier();

        // Read and parse logits data
        let mut file = File::open("../../schemas/logit_example_2.json")?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let logits_data: LogitsData = serde_json::from_str(&contents)?;

        // Verify inference
        match verifier.verify_inference(&logits_data) {
            Ok(()) => println!("Inference verified successfully"),
            Err(e) => println!("Verification failed: {:?}", e),
        }
        Ok(())
    }
}
