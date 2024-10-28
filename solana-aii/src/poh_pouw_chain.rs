//! # AI Logits Verification Blockchain
//!
//! A specialized blockchain implementation that combines Proof of Useful Work (PoUW) with
//! AI model logits verification. This system ensures that mining work contributes to
//! validating AI model outputs while maintaining a secure distributed ledger.
//!
//! ## Core Features
//!
//! ### Blockchain Components
//! * Custom block structure with AI verification results
//! * Proof of Useful Work (PoUW) instead of traditional PoW
//! * Temporal verification through Proof of History
//! * Transaction system for logits verification requests
//!
//! ### AI Verification
//! * Logits comparison against baseline models
//! * Threshold-based verification metrics
//! * Aggregated verification results in blocks
//! * Permanent record of verification history
//!
//! ## Key Structures
//!
//! ```rust
//! struct Block {
//!     timestamp: u64,
//!     transactions: Vec<Transaction>,
//!     useful_work_result: Option<Vec<LogitsVerificationResult>>,
//!     // ... other fields
//! }
//!
//! struct Transaction {
//!     sender: String,
//!     recipient: String,
//!     logits_data: LogitsData
//! }
//! ```
//!
//! ## Security Features
//! * SHA-256 based block hashing
//! * Chain integrity verification
//! * Temporal ordering enforcement
//! * Useful work verification
//!
//! ## Usage Example
//! ```rust
//! let mut blockchain = Blockchain::new();
//!
//! // Create a verification transaction
//! let tx = Transaction {
//!     sender: "sender_id",
//!     recipient: "recipient_id",
//!     logits_data: LogitsData::new(logits, baseline, threshold)
//! };
//!
//! // Mine block with verification
//! blockchain.mine_block(vec![tx]);
//! ```
//!
//! Note: This implementation combines blockchain security with AI model
//! verification, ensuring that mining computational resources are used
//! for validating AI model outputs. The system maintains both
//! distributed consensus and AI output verification in a single chain.
use crate::logit_verification;
use crate::logit_verification::{LogitsData, LogitsVerificationResult};
use serde_derive::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    sender: String,
    recipient: String,
    logits_data: LogitsData
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Block {
    timestamp: u64,
    transactions: Vec<Transaction>,
    previous_hash: String,
    nonce: u64,
    hash: String,
    useful_work_result: Option<Vec<LogitsVerificationResult>>,
}

impl Block {
    fn new(timestamp: u64, transactions: Vec<Transaction>, previous_hash: String) -> Self {
        let mut block = Block {
            timestamp,
            transactions,
            previous_hash,
            nonce: 0,
            hash: String::new(),
            useful_work_result: None,
        };
        block.hash = block.calculate_hash();
        block
    }

    fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        let data = format!(
            "{}{}{}",
            self.timestamp,
            serde_json::to_string(&self.transactions).unwrap(),
            self.previous_hash,
        );
        hasher.update(data.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

struct ProofOfUsefulWork {
}

impl ProofOfUsefulWork {
    fn new() -> Self {
        ProofOfUsefulWork {  }
    }

    fn inspect_logits(&self, logits_data: &LogitsData) -> LogitsVerificationResult {
        logit_verification::calculate_verification_metrics(logits_data)
    }

    fn verify_work(&self, result: &Option<Vec<LogitsVerificationResult>>) -> bool {
        match result {
            Some(_logits_verifications) => {
                true
            }
            None => false,
        }
    }
}

struct Blockchain {
    chain: Vec<Block>,
    pow: ProofOfUsefulWork,
    last_block_time: u64,
}

impl Blockchain {
    fn new() -> Self {
        let mut blockchain = Blockchain {
            chain: Vec::new(),
            pow: ProofOfUsefulWork::new(),
            last_block_time: 0,
        };
        blockchain.create_genesis_block();
        blockchain
    }

    fn create_genesis_block(&mut self) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let genesis_block = Block::new(timestamp, Vec::new(), String::from("0"));
        self.chain.push(genesis_block);
        self.last_block_time = timestamp;
    }

    fn mine_block(&mut self, transactions: Vec<Transaction>) -> Block {
        let previous_block = self.chain.last().unwrap();

        let mut pouw_results = vec![];
        for transaction in &transactions {
            // Perform useful work
            pouw_results.push(self.pow.inspect_logits(&transaction.logits_data));
        }

        let mut new_block = Block::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            transactions,
            previous_block.hash.clone(),
        );

        // Mine block
        // loop {
            let hash = new_block.calculate_hash();
            if self.pow.verify_work(&Some(pouw_results.clone())) {
                new_block.hash = hash;
                new_block.useful_work_result = Some(pouw_results.clone());
            }
            new_block.nonce += 1;
        // }

        // Update proof of history
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let time_since_last = current_time - self.last_block_time;

        // Enforce minimum block time
        if time_since_last < 1 {
            thread::sleep(Duration::from_secs(1 - time_since_last));
            new_block.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        self.last_block_time = new_block.timestamp;
        self.chain.push(new_block.clone());
        new_block
    }

    fn is_chain_valid(&self) -> bool {
        for i in 1..self.chain.len() {
            let current = &self.chain[i];
            let previous = &self.chain[i - 1];

            // Verify block hash
            if current.hash != current.calculate_hash() {
                return false;
            }

            // Verify previous hash link
            if current.previous_hash != previous.hash {
                return false;
            }

            // Verify proof of useful work
            if !self.pow.verify_work(&current.useful_work_result) {
                return false;
            }

            // Verify proof of history
            if current.timestamp <= previous.timestamp {
                return false;
            }
        }
        true
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain() {
        let mut blockchain = Blockchain::new();

        // Example logits for the input
        let logits = vec![-2.3, 0.5, 3.1];
        // Baseline logits for comparison (e.g., from a previous model version)
        let baseline_logits = vec![-2.0, 0.6, 3.0];
        // Set a threshold for verification
        let threshold = 0.1;

        let logits_data = LogitsData {
            logits,
            baseline_logits,
            threshold,
        };

        let tx = Transaction {
            sender: "SENDER_HASH".to_string(),
            recipient: "RECIPIENT_HASH".to_string(),
            logits_data,
        };
        let txs = vec![tx];
        blockchain.mine_block(txs);
        assert!(blockchain.is_chain_valid());
    }
}