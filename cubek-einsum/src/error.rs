//! Error types for einsum operations.

use alloc::string::String;

/// Errors that can occur during einsum parsing and execution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum EinsumError {
    /// Invalid einsum notation syntax.
    #[cfg_attr(feature = "std", error("parse error: {message}"))]
    ParseError { message: String },

    /// Index appears in output but not in any input.
    #[cfg_attr(feature = "std", error("output index '{index}' not found in any input"))]
    OutputIndexNotInInputs { index: char },

    /// Index appears more than twice (not a valid contraction).
    #[cfg_attr(feature = "std", error("index '{index}' appears {count} times, maximum is 2"))]
    IndexAppearsMoreThanTwice { index: char, count: usize },

    /// Inconsistent ellipsis usage.
    #[cfg_attr(feature = "std", error("inconsistent ellipsis: {message}"))]
    InconsistentEllipsis { message: String },

    /// Shape mismatch for a contracted index.
    #[cfg_attr(feature = "std", error("shape mismatch for index '{index}': expected {expected}, got {got}"))]
    ShapeMismatch {
        index: char,
        expected: usize,
        got: usize,
    },

    /// Incompatible number of dimensions.
    #[cfg_attr(feature = "std", error("dimension mismatch: subscript '{subscript}' expects {expected} dims, tensor has {got}"))]
    DimensionMismatch {
        subscript: String,
        expected: usize,
        got: usize,
    },

    /// Ellipsis dimension count mismatch.
    #[cfg_attr(feature = "std", error("ellipsis dimension mismatch: tensors have different batch dimensions"))]
    EllipsisDimensionMismatch {
        expected: usize,
        got: usize,
    },

    /// Empty subscript where one is required.
    #[cfg_attr(feature = "std", error("empty subscript not allowed"))]
    EmptySubscript,

    /// No inputs provided.
    #[cfg_attr(feature = "std", error("at least one input tensor is required"))]
    NoInputs,

    /// Kernel launch error.
    #[cfg_attr(feature = "std", error("launch error: {message}"))]
    LaunchError { message: String },

    /// Unsupported operation.
    #[cfg_attr(feature = "std", error("unsupported operation: {message}"))]
    Unsupported { message: String },

    /// Memory allocation error.
    #[cfg_attr(feature = "std", error("memory error: {message}"))]
    MemoryError { message: String },

    /// Shape computation error.
    #[cfg_attr(feature = "std", error("shape error: {message}"))]
    ShapeError { message: String },
}

impl EinsumError {
    pub fn parse(message: impl Into<String>) -> Self {
        Self::ParseError {
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    pub fn launch(message: impl Into<String>) -> Self {
        Self::LaunchError {
            message: message.into(),
        }
    }

    pub fn memory(message: impl Into<String>) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    pub fn shape(message: impl Into<String>) -> Self {
        Self::ShapeError {
            message: message.into(),
        }
    }
}

/// Result type for einsum operations.
pub type EinsumResult<T> = core::result::Result<T, EinsumError>;
