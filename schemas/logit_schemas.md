## Schema 1

[Logit Schema 1](logit_schema.json)

This schema, labeled as `LogitDataModel`, is structured to log and verify AI inference data, specifically focusing on logits (raw model output before softmax normalization) and metrics for monitoring model reliability. Here’s a breakdown of the schema elements and how they relate to inference verification from an AI logits perspective:

### 1. **Input Block**
- **Purpose:** Captures the input data fed into the model, which is essential for tracking and re-verifying inference outputs.
- **Key Components:**
    - **input_id:** Unique identifier for each input, crucial for linking back to specific inferences.
    - **input_data:** The raw input data, which could be text, images, or any data type used by the model, allowing traceability.
    - **timestamp:** Records when the input was processed, useful for chronological analysis and drift monitoring.
    - **source:** Identifies the origin of the input, allowing context-based analysis and helping in categorizing or filtering inference sources.

### 2. **Model Metadata Block**
- **Purpose:** Stores metadata about the model that produced the logits, useful for version control and environment-specific evaluation.
- **Key Components:**
    - **model_id and version:** Identify the specific model and its version, essential for comparing outputs across versions and understanding changes over time.
    - **model_type:** Describes the nature of the model (e.g., classifier, regressor), providing context for how logits should be interpreted.
    - **environment:** Specifies the environment in which the model operates (e.g., production, staging), which impacts inference reliability.

### 3. **Logits Block**
- **Purpose:** Contains the logits output and related data that are used to verify predictions.
- **Key Components:**
    - **class_names:** An array of possible classes the model can predict. These should align with the output dimensions in the logits array.
    - **values:** The raw logits corresponding to each class in `class_names`, which provide an unnormalized measure of the model’s confidence across classes.
    - **predicted_class:** The class with the highest logit value, representing the model’s final output.
    - **confidence_score:** A normalized measure of confidence (usually derived from the logits using softmax), helpful for understanding the model’s certainty in its prediction.

- **Relevance for Inference Verification:** By capturing both logits and the predicted class, this structure supports post-inference evaluations, such as checking if high logits correlate with expected predictions and analyzing confidence consistency. It enables re-interpretation of logits in case of post-hoc adjustments to model thresholds or for debugging.

### 4. **Verification Metrics Block**
- **Purpose:** Provides quantitative metrics to evaluate model output quality, robustness, and detect anomalies.
- **Key Components:**
    - **cosine_similarity_score:** Measures similarity between logits and a reference, helping detect if current outputs deviate from the norm.
    - **drift_score:** Captures model drift, possibly calculated against a baseline or historical logits, which can signal if the model's inference behavior changes over time.
    - **threshold_violation_flag:** A boolean that indicates if any verification threshold (e.g., confidence or similarity threshold) is breached, flagging potential prediction reliability issues.

- **Relevance for Inference Verification:** These metrics serve as a safeguard, flagging outputs that may not be reliable or may indicate a shift in model performance, helping catch inference anomalies early.

### Overall Schema Use
In AI inference verification, this schema provides a comprehensive structure to monitor and analyze logits-based outputs. By logging model version, input data, raw logits, and verification metrics, it ensures that predictions are traceable and that each aspect of the inference can be cross-checked for consistency, accuracy, and stability over time.

## Schema 2

[Logit Schema 2](logit_schema_2.json)

This `AI Model Logits Schema` is designed to capture AI model inference data with a focus on the output logits, parameters affecting generation, and data integrity checks. Here's a breakdown of its key components and a comparison to the previous `LogitDataModel` schema:

### Structure Overview
1. **Model Identification and Timing**
    - **model_id** and **timestamp**: Identify the AI model and timestamp the logits were generated. Similar to `LogitDataModel`, these provide versioning, traceability, and chronological ordering.

2. **Logits Data Block**
    - **sequence_length** and **vocabulary_size**: Define the dimensions of the logits output.
    - **logits**:
        - **values**: Contains a 2D array of raw logits values in the format [sequence_length x vocabulary_size], allowing for sequence-based model outputs.
        - **shape**: Specifies the logits matrix dimensions for verification.
    - **Comparison**: In contrast to the previous schema, this logits block focuses on models generating sequences (e.g., language models), whereas `LogitDataModel` targeted a simple classification logits array.

3. **Metadata Block**
    - **Sampling Parameters (temperature, top_k, top_p)**: Detail the inference parameters used for generating predictions.
    - **input_text** and **token_ids**: Include the raw input text and its corresponding token IDs, giving context for decoding and re-generating logits if necessary.
    - **Comparison**: The `LogitDataModel` included more general metadata on the model environment (production, staging, etc.) rather than sampling parameters specific to probabilistic outputs. This schema targets fine-tuning generation parameters, which is particularly useful for language models or generative AI systems.

4. **Verification Block**
    - **Hashing and Digital Signatures**: Adds a verification layer for data integrity and authenticity, capturing:
        - **hash_algorithm**: Specifies the cryptographic hash function (e.g., SHA-256).
        - **hash_input_fields** and **hash**: Define fields used to calculate the hash, providing a fingerprint for verifying data consistency.
        - **signature** and **certificate**: Allow for digital signatures, helping verify the source.
    - **Comparison**: The previous schema’s verification focused on inference reliability (cosine similarity, drift), whereas this schema emphasizes data integrity and origin verification through cryptographic methods.

### Comparison Summary
| Feature                 | Schema 1                                            | Schema 2                                                                                          |
|-------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Model ID and Timestamp** | Tracks model and timing information                 | Tracks model ID and timestamp with similar structure                                              |
| **Input Structure**     | Captures input data, source, and timestamp          | Captures sequence and vocabulary size, reflecting a sequence-based model                          |
| **Logits Representation** | Single logit array for classification or regression | 2D logits array with sequence and vocabulary dimensions, suited for sequence generation           |
| **Metadata**            | Model environment and ID                            | Sampling parameters (temperature, top-k, top-p), input text, token IDs for generation specificity |
| **Verification**        | Inference verification (cosine similarity, drift)   | Data integrity (hashing, signatures, certificates) for secure logging and tamper prevention       |

### Usage Context
- **LogitDataModel**: Geared towards classification or single inference tasks with real-time monitoring metrics like drift detection and confidence scoring.
- **AI Model Logits Schema**: Optimized for generative models, especially sequence-based (e.g., language models), with cryptographic security for data integrity. Its emphasis on sampling parameters and sequence structure is ideal for text generation contexts, where probabilistic outputs depend on fine-tuned settings.