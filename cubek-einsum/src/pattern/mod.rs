//! Pattern recognition for einsum fast paths.
//!
//! Recognizes common operations that can be dispatched to optimized kernels:
//! - Matrix multiplication
//! - Batched matrix multiplication
//! - Transpose
//! - Reduction (sum, prod, max, min)
//! - Hadamard (element-wise) product
//! - Outer product
//! - Dot product
//! - Trace

mod fast_path;
mod matmul;
mod unary;
mod binary;

pub use fast_path::FastPath;
pub use matmul::{is_matmul, is_batched_matmul, extract_matmul_config, MatmulConfig};
pub use unary::{is_transpose, is_reduction, is_trace, is_diagonal_extract};
pub use binary::{is_hadamard, is_outer_product, is_dot_product};

use crate::notation::EinsumNotation;

/// Pattern matcher for einsum operations.
pub struct PatternMatcher;

impl PatternMatcher {
    /// Attempts to recognize a fast-path pattern.
    pub fn recognize(notation: &EinsumNotation) -> Option<FastPath> {
        recognize_pattern(notation)
    }
}

/// Main entry point for pattern recognition.
///
/// Tries to match the notation against known patterns in order of specificity.
pub fn recognize_pattern(notation: &EinsumNotation) -> Option<FastPath> {
    // Unary operations
    if notation.is_unary() {
        // Check transpose first (most common)
        if let Some(perm) = is_transpose(notation) {
            return Some(FastPath::Transpose { permutation: perm });
        }

        // Check trace
        if is_trace(notation) {
            return Some(FastPath::Trace);
        }

        // Check diagonal extraction
        if is_diagonal_extract(notation).is_some() {
            return Some(FastPath::DiagonalExtract);
        }

        // Check reduction
        if let Some((axes, _op)) = is_reduction(notation) {
            return Some(FastPath::Reduce { axes, op: fast_path::ReduceOp::Sum });
        }
    }

    // Binary operations
    if notation.is_binary() {
        // Check for batched matmul first (more specific)
        if let Some(config) = is_batched_matmul(notation) {
            return Some(FastPath::BatchedMatmul {
                batch_dims: config.batch_dims,
                transpose_a: config.transpose_a,
                transpose_b: config.transpose_b,
            });
        }

        // Check for regular matmul
        if let Some(config) = is_matmul(notation) {
            return Some(FastPath::Matmul {
                transpose_a: config.transpose_a,
                transpose_b: config.transpose_b,
            });
        }

        // Check Hadamard product
        if is_hadamard(notation) {
            return Some(FastPath::Hadamard);
        }

        // Check outer product
        if is_outer_product(notation) {
            return Some(FastPath::OuterProduct);
        }

        // Check dot product
        if is_dot_product(notation) {
            return Some(FastPath::DotProduct);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_recognize_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::Matmul { .. })));
    }

    #[test]
    fn test_recognize_batched_matmul() {
        let notation = parse_einsum("bij,bjk->bik").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::BatchedMatmul { .. })));
    }

    #[test]
    fn test_recognize_transpose() {
        let notation = parse_einsum("ij->ji").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::Transpose { .. })));
    }

    #[test]
    fn test_recognize_hadamard() {
        let notation = parse_einsum("ij,ij->ij").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::Hadamard)));
    }

    #[test]
    fn test_recognize_outer_product() {
        let notation = parse_einsum("i,j->ij").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::OuterProduct)));
    }

    #[test]
    fn test_recognize_dot_product() {
        let notation = parse_einsum("i,i->").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::DotProduct)));
    }

    #[test]
    fn test_recognize_trace() {
        let notation = parse_einsum("ii->").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::Trace)));
    }

    #[test]
    fn test_recognize_reduction() {
        let notation = parse_einsum("ij->i").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(matches!(pattern, Some(FastPath::Reduce { .. })));
    }

    #[test]
    fn test_no_pattern_for_complex() {
        let notation = parse_einsum("ijk,jkl,klm->im").unwrap();
        let pattern = recognize_pattern(&notation);

        assert!(pattern.is_none());
    }
}
