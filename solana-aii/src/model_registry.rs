//! # AI Model Registry
//!
//! A Solana-based registry system for managing machine learning models, their versions,
//! and deployment configurations. This system provides functionality for:
//! - Model registration and versioning
//! - Version verification and validation
//! - Access control and permissions
//! - Model forking and rollbacks
//! - Deployment configuration management
use borsh::{BorshDeserialize, BorshSerialize};
use solana_program::pubkey::Pubkey;
use solana_program::program_error::ProgramError;
use solana_program::entrypoint::ProgramResult;
use std::collections::BTreeMap;

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct ComputeSpec {
    pub memory_requirement: u64,      // in bytes
    pub cpu_requirement: u64,         // in compute units
    pub gpu_requirement: Option<u64>, // optional GPU requirements
    pub storage_requirement: u64,     // in bytes
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct NormalizationParams {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct InputConstraint {
    pub constraint_type: InputConstraintType,
    pub parameters: Vec<f64>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum InputConstraintType {
    Range,
    Categorical,
    Pattern,
    Custom(String),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct LogitsSpec {
    pub temperature: f64,
    pub top_k: Option<u32>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum OutputInterpretation {
    Classification {
        classes: Vec<String>,
        threshold: Option<f64>,
    },
    Regression {
        units: String,
        range: (f64, f64),
    },
    Embedding {
        dimension: usize,
        metric: String,
    },
    Generation {
        max_length: usize,
        stop_tokens: Vec<String>,
    },
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub latency: u64,             // in milliseconds
    pub throughput: u64,          // requests per second
    pub error_rate: f64,
    pub custom_metrics: Vec<(String, f64)>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum VerificationStatus {
    Pending,
    InProgress,
    Verified,
    Failed(String),
    Archived(ArchiveReason),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum ArchiveReason {
    Deprecated,
    SecurityVulnerability,
    Performance,
    Superseded,
    Other(String),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct VerificationProof {
    pub verifier: Pubkey,
    pub timestamp: i64,
    pub score: u64,
    pub metrics: ModelMetrics,
    pub verification_data: Vec<u8>,
    pub signature: [u8; 64],
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct ForkProof {
    pub original_version: u32,
    pub fork_reason: String,
    pub compatibility_proof: Vec<u8>,
    pub authorizer: Pubkey,
    pub timestamp: i64,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct RollbackProof {
    pub reason: String,
    pub impact_assessment: String,
    pub approver: Pubkey,
    pub timestamp: i64,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct ModelLicense {
    pub license_type: String,
    pub terms: String,
    pub permissions: Vec<String>,
    pub restrictions: Vec<String>,
    pub attribution_required: bool,
}

// Add Default implementation for ModelVersion as it's used in fork_model
impl Default for ModelVersion {
    fn default() -> Self {
        Self {
            version_number: 0,
            weights_merkle_root: [0; 32],
            architecture_hash: [0; 32],
            timestamp: 0,
            deployer: Pubkey::default(),
            verification_status: VerificationStatus::Pending,
            performance_metrics: ModelMetrics {
                accuracy: 0.0,
                latency: 0,
                throughput: 0,
                error_rate: 0.0,
                custom_metrics: Vec::new(),
            },
            previous_version: None,
            upgrade_proof: None,
        }
    }
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct ModelVersion {
    version_number: u32,
    weights_merkle_root: [u8; 32],
    architecture_hash: [u8; 32],
    timestamp: i64,
    deployer: Pubkey,
    verification_status: VerificationStatus,
    performance_metrics: ModelMetrics,
    previous_version: Option<Pubkey>,
    upgrade_proof: Option<UpgradeProof>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct ModelMetadata {
    name: String,
    description: String,
    architecture: ModelArchitecture,
    license: ModelLicense,
    input_spec: InputSpecification,
    output_spec: OutputSpecification,
    current_version: u32,
    versions: BTreeMap<u32, ModelVersion>,
    access_control: AccessControl,
    deployment_config: DeploymentConfig,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct ModelArchitecture {
    framework: String,            // e.g., "transformer", "cnn", "mlp"
    layer_config: Vec<LayerSpec>,
    total_parameters: u64,
    computation_requirements: ComputeSpec,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct LayerSpec {
    layer_type: String,
    input_dim: Vec<usize>,
    output_dim: Vec<usize>,
    activation: String,
    parameters: u64,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct InputSpecification {
    shape: Vec<usize>,
    dtype: String,
    normalization: Option<NormalizationParams>,
    constraints: Vec<InputConstraint>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct OutputSpecification {
    shape: Vec<usize>,
    dtype: String,
    logits_spec: LogitsSpec,
    interpretation: OutputInterpretation,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct AccessControl {
    owners: Vec<Pubkey>,
    authorized_deployers: Vec<Pubkey>,
    authorized_verifiers: Vec<Pubkey>,
    access_policy: AccessPolicy,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum AccessPolicy {
    Public,
    VerifierOnly,
    OwnerOnly,
    WhitelistedUsers(Vec<Pubkey>),
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct DeploymentConfig {
    min_verifier_score: u64,
    required_verifications: u8,
    max_verification_time: u64,
    compute_requirements: ComputeSpec,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub struct UpgradeProof {
    old_weights_hash: [u8; 32],
    new_weights_hash: [u8; 32],
    upgrade_type: UpgradeType,
    verification_proofs: Vec<VerificationProof>,
}

#[derive(BorshSerialize, BorshDeserialize, Debug, Clone)]
pub enum UpgradeType {
    WeightUpdate,
    ArchitectureChange,
    BugFix,
    SecurityPatch,
}

// Main registry implementation
pub struct ModelRegistry {
    models: BTreeMap<Pubkey, ModelMetadata>,
    version_history: BTreeMap<Pubkey, Vec<ModelVersion>>,
}

impl ModelRegistry {
    pub fn register_model(
        &mut self,
        model_key: &Pubkey,
        metadata: ModelMetadata,
        initial_version: ModelVersion,
    ) -> ProgramResult {
        // Validate metadata and version
        self.validate_new_model(&metadata, &initial_version)?;

        // Initialize version history
        let mut versions = Vec::new();
        versions.push(initial_version);

        // Store model and version history
        self.models.insert(*model_key, metadata);
        self.version_history.insert(*model_key, versions);

        Ok(())
    }

    pub fn upgrade_model(
        &mut self,
        model_key: &Pubkey,
        new_version: ModelVersion,
        upgrade_proof: UpgradeProof,
    ) -> ProgramResult {
        // Validate upgrade
        self.validate_upgrade(model_key, &new_version, &upgrade_proof)?;

        let metadata = self.models.get_mut(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        // Update version history
        let versions = self.version_history.get_mut(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;
        versions.push(new_version);

        // Update metadata
        metadata.current_version += 1;

        Ok(())
    }

    pub fn verify_version(
        &mut self,
        model_key: &Pubkey,
        version: u32,
        verification: VerificationProof,
    ) -> ProgramResult {
        // Validate verification proof
        self.validate_verification(&verification)?;

        let versions = self.version_history.get_mut(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        let version_data = versions.iter_mut()
            .find(|v| v.version_number == version)
            .ok_or(ProgramError::InvalidAccountData)?;

        // Update version verification status
        version_data.verification_status = VerificationStatus::Verified;

        Ok(())
    }

    pub fn get_version_info(
        &self,
        model_key: &Pubkey,
        version: u32,
    ) -> Result<&ModelVersion, ProgramError> {
        let versions = self.version_history.get(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        versions.iter()
            .find(|v| v.version_number == version)
            .ok_or(ProgramError::InvalidAccountData)
    }

    // Fork a model to create a new variant
    pub fn fork_model(
        &mut self,
        original_key: &Pubkey,
        new_key: &Pubkey,
        fork_metadata: ModelMetadata,
        fork_proof: ForkProof,
    ) -> ProgramResult {
        // Validate fork permissions and proof
        self.validate_fork(original_key, &fork_metadata, &fork_proof)?;

        // Create new model entry with fork history
        let initial_version = ModelVersion {
            version_number: 1,
            previous_version: Some(*original_key),
            ..Default::default()
        };

        self.register_model(new_key, fork_metadata, initial_version)
    }

    // Rollback to a previous version
    pub fn rollback_version(
        &mut self,
        model_key: &Pubkey,
        target_version: u32,
        rollback_proof: RollbackProof,
    ) -> ProgramResult {
        // Validate rollback
        self.validate_rollback(model_key, target_version, &rollback_proof)?;

        let metadata = self.models.get_mut(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        // Update current version
        metadata.current_version = target_version;

        Ok(())
    }

    // Archive or deprecate a model version
    pub fn archive_version(
        &mut self,
        model_key: &Pubkey,
        version: u32,
        archive_reason: ArchiveReason,
    ) -> ProgramResult {
        let versions = self.version_history.get_mut(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        if let Some(version_data) = versions.iter_mut()
            .find(|v| v.version_number == version) {
            version_data.verification_status = VerificationStatus::Archived(archive_reason);
        }

        Ok(())
    }

    // Validate a new model being registered
    fn validate_new_model(
        &self,
        metadata: &ModelMetadata,
        initial_version: &ModelVersion,
    ) -> ProgramResult {
        // Check basic metadata requirements
        if metadata.name.is_empty() || metadata.description.is_empty() {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate architecture configuration
        if metadata.architecture.layer_config.is_empty()
            || metadata.architecture.total_parameters == 0 {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate input/output specifications
        if metadata.input_spec.shape.is_empty()
            || metadata.output_spec.shape.is_empty() {
            return Err(ProgramError::InvalidArgument);
        }

        // Ensure at least one owner is specified
        if metadata.access_control.owners.is_empty() {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate deployment config
        if metadata.deployment_config.required_verifications == 0
            || metadata.deployment_config.max_verification_time == 0 {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate initial version
        if initial_version.version_number != 1 {
            return Err(ProgramError::InvalidArgument);
        }

        Ok(())
    }

    // Validate model upgrade
    fn validate_upgrade(
        &self,
        model_key: &Pubkey,
        new_version: &ModelVersion,
        upgrade_proof: &UpgradeProof,
    ) -> ProgramResult {
        // Get current model metadata
        let metadata = self.models.get(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        // Ensure version number is sequential
        if new_version.version_number != metadata.current_version + 1 {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate upgrade proof
        match upgrade_proof.upgrade_type {
            UpgradeType::WeightUpdate => {
                // Ensure architecture hasn't changed
                if new_version.architecture_hash != self.get_latest_version(model_key)?.architecture_hash {
                    return Err(ProgramError::InvalidArgument);
                }
            }
            UpgradeType::ArchitectureChange => {
                // Require additional verification proofs for architecture changes
                if upgrade_proof.verification_proofs.len() < 2 {
                    return Err(ProgramError::InvalidArgument);
                }
            }
            _ => {}
        }

        // Validate all verification proofs
        for proof in &upgrade_proof.verification_proofs {
            self.validate_verification(proof)?;
        }

        Ok(())
    }

    // Get the latest version of a model
    fn get_latest_version(&self, model_key: &Pubkey) -> Result<&ModelVersion, ProgramError> {
        let versions = self.version_history.get(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        versions.last()
            .ok_or(ProgramError::InvalidAccountData)
    }

    // Validate a verification proof
    fn validate_verification(&self, verification: &VerificationProof) -> ProgramResult {
        // Check if verifier is authorized
        // Note: You would need to implement actual signature verification here
        if verification.signature == [0; 64] {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate metrics
        if verification.metrics.accuracy < 0.0
            || verification.metrics.accuracy > 1.0
            || verification.metrics.error_rate < 0.0 {
            return Err(ProgramError::InvalidArgument);
        }

        // Ensure verification data is present
        if verification.verification_data.is_empty() {
            return Err(ProgramError::InvalidArgument);
        }

        Ok(())
    }

    // Validate model fork
    fn validate_fork(
        &self,
        original_key: &Pubkey,
        fork_metadata: &ModelMetadata,
        fork_proof: &ForkProof,
    ) -> ProgramResult {
        // Ensure original model exists
        let original_metadata = self.models.get(original_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        // Check if fork is allowed based on license
        if fork_metadata.license.restrictions.contains(&"no-fork".to_string()) {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate fork proof
        if fork_proof.compatibility_proof.is_empty() {
            return Err(ProgramError::InvalidArgument);
        }

        // Ensure fork has a valid reason
        if fork_proof.fork_reason.is_empty() {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate authorizer is in original model's owners or has fork permissions
        if !original_metadata.access_control.owners.contains(&fork_proof.authorizer) {
            return Err(ProgramError::InvalidArgument);
        }

        Ok(())
    }

    // Validate version rollback
    fn validate_rollback(
        &self,
        model_key: &Pubkey,
        target_version: u32,
        rollback_proof: &RollbackProof,
    ) -> ProgramResult {
        // Get current model metadata and versions
        let metadata = self.models.get(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        let versions = self.version_history.get(model_key)
            .ok_or(ProgramError::InvalidAccountData)?;

        // Ensure target version exists and is less than current version
        if target_version >= metadata.current_version
            || !versions.iter().any(|v| v.version_number == target_version) {
            return Err(ProgramError::InvalidArgument);
        }

        // Validate rollback reason and impact assessment
        if rollback_proof.reason.is_empty()
            || rollback_proof.impact_assessment.is_empty() {
            return Err(ProgramError::InvalidArgument);
        }

        // Ensure approver has necessary permissions
        if !metadata.access_control.owners.contains(&rollback_proof.approver) {
            return Err(ProgramError::InvalidArgument);
        }

        Ok(())
    }

    // Create a new empty registry
    pub fn new() -> Self {
        Self {
            models: BTreeMap::new(),
            version_history: BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solana_program::program_error::ProgramError;

    fn create_test_compute_spec() -> ComputeSpec {
        ComputeSpec {
            memory_requirement: 1024 * 1024 * 1024, // 1GB
            cpu_requirement: 1000,
            gpu_requirement: Some(2000),
            storage_requirement: 5 * 1024 * 1024 * 1024, // 5GB
        }
    }

    fn create_test_layer_spec() -> LayerSpec {
        LayerSpec {
            layer_type: "linear".to_string(),
            input_dim: vec![784],
            output_dim: vec![512],
            activation: "relu".to_string(),
            parameters: 401920,
        }
    }

    fn create_test_model_architecture() -> ModelArchitecture {
        ModelArchitecture {
            framework: "transformer".to_string(),
            layer_config: vec![create_test_layer_spec()],
            total_parameters: 401920,
            computation_requirements: create_test_compute_spec(),
        }
    }

    fn create_test_input_spec() -> InputSpecification {
        InputSpecification {
            shape: vec![1, 28, 28],
            dtype: "float32".to_string(),
            normalization: Some(NormalizationParams {
                mean: vec![0.5],
                std: vec![0.5],
                min_value: Some(0.0),
                max_value: Some(1.0),
            }),
            constraints: vec![],
        }
    }

    fn create_test_output_spec() -> OutputSpecification {
        OutputSpecification {
            shape: vec![10],
            dtype: "float32".to_string(),
            logits_spec: LogitsSpec {
                temperature: 1.0,
                top_k: Some(5),
                top_p: Some(0.9),
                repetition_penalty: Some(1.2),
            },
            interpretation: OutputInterpretation::Classification {
                classes: vec!["0".to_string(), "1".to_string(), "2".to_string()],
                threshold: Some(0.5),
            },
        }
    }

    fn create_test_access_control() -> AccessControl {
        AccessControl {
            owners: vec![Pubkey::new_unique()],
            authorized_deployers: vec![Pubkey::new_unique()],
            authorized_verifiers: vec![Pubkey::new_unique()],
            access_policy: AccessPolicy::Public,
        }
    }

    fn create_test_deployment_config() -> DeploymentConfig {
        DeploymentConfig {
            min_verifier_score: 80,
            required_verifications: 2,
            max_verification_time: 3600,
            compute_requirements: create_test_compute_spec(),
        }
    }

    fn create_test_model_metadata() -> ModelMetadata {
        ModelMetadata {
            name: "Test Model".to_string(),
            description: "A test model for unit testing".to_string(),
            architecture: create_test_model_architecture(),
            license: ModelLicense {
                license_type: "MIT".to_string(),
                terms: "Test terms".to_string(),
                permissions: vec!["commercial-use".to_string()],
                restrictions: vec![],
                attribution_required: true,
            },
            input_spec: create_test_input_spec(),
            output_spec: create_test_output_spec(),
            current_version: 1,
            versions: BTreeMap::new(),
            access_control: create_test_access_control(),
            deployment_config: create_test_deployment_config(),
        }
    }

    fn create_test_model_version() -> ModelVersion {
        ModelVersion {
            version_number: 1,
            weights_merkle_root: [1; 32],
            architecture_hash: [1; 32],
            timestamp: 1635724800,
            deployer: Pubkey::new_unique(),
            verification_status: VerificationStatus::Pending,
            performance_metrics: ModelMetrics {
                accuracy: 0.95,
                latency: 100,
                throughput: 1000,
                error_rate: 0.05,
                custom_metrics: vec![],
            },
            previous_version: None,
            upgrade_proof: None,
        }
    }

    #[test]
    fn test_register_model() {
        let mut registry = ModelRegistry::new();
        let model_key = Pubkey::new_unique();
        let metadata = create_test_model_metadata();
        let version = create_test_model_version();

        assert!(registry.register_model(&model_key, metadata, version).is_ok());

        // Test duplicate registration
        let metadata2 = create_test_model_metadata();
        let version2 = create_test_model_version();
        assert!(registry.register_model(&model_key, metadata2, version2).is_ok()); // Should overwrite
    }

    #[test]
    fn test_register_model_invalid() {
        let mut registry = ModelRegistry::new();
        let model_key = Pubkey::new_unique();
        let mut metadata = create_test_model_metadata();
        let version = create_test_model_version();

        // Test empty name
        metadata.name = "".to_string();
        assert_eq!(
            registry.register_model(&model_key, metadata.clone(), version.clone()),
            Err(ProgramError::InvalidArgument)
        );
    }

    #[test]
    fn test_upgrade_model() {
        let mut registry = ModelRegistry::new();
        let model_key = Pubkey::new_unique();

        // First register the model
        let metadata = create_test_model_metadata();
        let initial_version = create_test_model_version();
        assert!(registry.register_model(&model_key, metadata, initial_version).is_ok());

        // Create upgrade version
        let mut new_version = create_test_model_version();
        new_version.version_number = 2;

        let upgrade_proof = UpgradeProof {
            old_weights_hash: [1; 32],
            new_weights_hash: [2; 32],
            upgrade_type: UpgradeType::WeightUpdate,
            verification_proofs: vec![],
        };

        let _ = registry.upgrade_model(&model_key, new_version, upgrade_proof).map_err(|err| {
            println!("{:?}", err);
            assert!(false)
        }).unwrap_or(assert!(true));
    }

    #[test]
    fn test_verify_version() {
        let mut registry = ModelRegistry::new();
        let model_key = Pubkey::new_unique();

        // Register model
        let metadata = create_test_model_metadata();
        let version = create_test_model_version();
        assert!(registry.register_model(&model_key, metadata, version).is_ok());

        // Create verification proof
        let verification = VerificationProof {
            verifier: Pubkey::new_unique(),
            timestamp: 1635724800,
            score: 95,
            metrics: ModelMetrics {
                accuracy: 0.95,
                latency: 100,
                throughput: 1000,
                error_rate: 0.05,
                custom_metrics: vec![],
            },
            verification_data: vec![1, 2, 3],
            signature: [1; 64],
        };

        assert!(registry.verify_version(&model_key, 1, verification).is_ok());
    }

    #[test]
    fn test_fork_model() {
        let mut registry = ModelRegistry::new();
        let original_key = Pubkey::new_unique();
        let new_key = Pubkey::new_unique();

        // Register original model
        let metadata = create_test_model_metadata();
        let version = create_test_model_version();
        assert!(registry.register_model(&original_key, metadata.clone(), version).is_ok());

        // Create fork proof
        let fork_proof = ForkProof {
            original_version: 1,
            fork_reason: "Testing new architecture".to_string(),
            compatibility_proof: vec![1, 2, 3],
            authorizer: metadata.access_control.owners[0],
            timestamp: 1635724800,
        };

        let _ = registry.fork_model(&original_key, &new_key, metadata, fork_proof).map_err(|err| {
            println!("{:?}", err);
            assert!(false)
        }).unwrap_or(assert!(true));
    }

    #[test]
    fn test_rollback_version() {
        let mut registry = ModelRegistry::new();
        let model_key = Pubkey::new_unique();

        // Register and upgrade model to have multiple versions
        let metadata = create_test_model_metadata();
        let initial_version = create_test_model_version();
        assert!(registry.register_model(&model_key, metadata, initial_version).is_ok());

        let mut new_version = create_test_model_version();
        new_version.version_number = 2;
        let upgrade_proof = UpgradeProof {
            old_weights_hash: [1; 32],
            new_weights_hash: [2; 32],
            upgrade_type: UpgradeType::WeightUpdate,
            verification_proofs: vec![],
        };

        let _ = registry.upgrade_model(&model_key, new_version, upgrade_proof).map_err(|err| {
            println!("{:?}", err);
            assert!(false)
        }).unwrap_or(assert!(true));

        // Create rollback proof
        let rollback_proof = RollbackProof {
            reason: "Performance regression".to_string(),
            impact_assessment: "Minimal impact expected".to_string(),
            approver: registry.models.get(&model_key).unwrap().access_control.owners[0],
            timestamp: 1635724800,
        };

        assert!(registry.rollback_version(&model_key, 1, rollback_proof).is_ok());
    }

    #[test]
    fn test_archive_version() {
        let mut registry = ModelRegistry::new();
        let model_key = Pubkey::new_unique();

        // Register model
        let metadata = create_test_model_metadata();
        let version = create_test_model_version();
        assert!(registry.register_model(&model_key, metadata, version).is_ok());

        assert!(registry.archive_version(&model_key, 1, ArchiveReason::Deprecated).is_ok());
    }
}