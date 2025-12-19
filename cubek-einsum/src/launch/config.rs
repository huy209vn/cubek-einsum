//! Configuration for einsum operations.

use crate::optimization::ContractionStrategy;

/// Configuration options for einsum execution.
#[derive(Debug, Clone)]
pub struct EinsumConfig {
    /// Strategy for finding contraction paths.
    pub strategy: ContractionStrategy,
    /// Whether to use tensor cores when available.
    pub use_tensor_cores: bool,
    /// Whether to enable autotuning.
    pub autotune: bool,
    /// Whether to validate shapes before execution.
    pub validate_shapes: bool,
}

impl Default for EinsumConfig {
    fn default() -> Self {
        Self {
            strategy: ContractionStrategy::Auto,
            use_tensor_cores: true,
            autotune: true,
            validate_shapes: true,
        }
    }
}

impl EinsumConfig {
    /// Creates a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the contraction strategy.
    pub fn with_strategy(mut self, strategy: ContractionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enables or disables tensor cores.
    pub fn with_tensor_cores(mut self, enabled: bool) -> Self {
        self.use_tensor_cores = enabled;
        self
    }

    /// Enables or disables autotuning.
    pub fn with_autotune(mut self, enabled: bool) -> Self {
        self.autotune = enabled;
        self
    }

    /// Enables or disables shape validation.
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validate_shapes = enabled;
        self
    }

    /// Creates a config optimized for speed (minimal validation).
    pub fn fast() -> Self {
        Self {
            strategy: ContractionStrategy::Greedy,
            use_tensor_cores: true,
            autotune: false,
            validate_shapes: false,
        }
    }

    /// Creates a config optimized for correctness (full validation).
    pub fn safe() -> Self {
        Self {
            strategy: ContractionStrategy::Optimal,
            use_tensor_cores: true,
            autotune: true,
            validate_shapes: true,
        }
    }
}
