use std::f64;
use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitsData {
    pub(crate) logits: Vec<f64>,
    pub(crate) baseline_logits: Vec<f64>,
    pub(crate) threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogitsVerificationResult {
    pub confidence: f64,
    pub cosine_sim: f64,
    pub drift: f64,
    pub threshold_violation: bool
}

// Softmax function to compute confidence score
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum_exp_logits).collect()
}

// Function to calculate confidence score (highest probability after softmax)
fn confidence_score(logits: &[f64]) -> f64 {
    let probabilities = softmax(logits);
    probabilities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
}

// Function to calculate cosine similarity between two logit vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot_product: f64 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let magnitude_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let magnitude_b: f64 = b.iter().map(|&y| y * y).sum::<f64>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}

// Function to calculate drift score (Mean Squared Error) between two logit vectors
fn drift_score(current_logits: &[f64], baseline_logits: &[f64]) -> f64 {
    current_logits
        .iter()
        .zip(baseline_logits)
        .map(|(&current, &baseline)| (current - baseline).powi(2))
        .sum::<f64>()
        / current_logits.len() as f64
}

// Main function to calculate verification metrics
pub(crate) fn calculate_verification_metrics(
    logits_data: &LogitsData,
) -> LogitsVerificationResult {
    let confidence = confidence_score(&logits_data.logits);
    let cosine_sim = cosine_similarity(&logits_data.logits, &logits_data.baseline_logits);
    let drift = drift_score(&logits_data.logits, &logits_data.baseline_logits);
    let threshold_violation = cosine_sim < logits_data.threshold || drift > logits_data.threshold;
    LogitsVerificationResult {
        confidence,
        cosine_sim,
        drift,
        threshold_violation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_metrics_and_threshold() {
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

        // Calculate verification metrics
        let result =
            calculate_verification_metrics(&logits_data);

        // Print results
        println!("Confidence Score: {:.4}", result.confidence);
        println!("Cosine Similarity: {:.4}", result.cosine_sim);
        println!("Drift Score: {:.4}", result.drift);
        println!("Threshold Violation: {}", result.threshold_violation);
    }
}
