//! Unary operation pattern detection.

use alloc::vec::Vec;
use alloc::collections::BTreeSet;

use crate::notation::EinsumNotation;

/// Checks if the notation represents a transpose (permutation).
///
/// Returns the permutation vector if it's a transpose.
/// `ij->ji` returns `[1, 0]`
pub fn is_transpose(notation: &EinsumNotation) -> Option<Vec<usize>> {
    if !notation.is_unary() {
        return None;
    }

    let input = &notation.inputs()[0];
    let output = notation.output();

    // All input indices must appear in output (no reduction)
    if !notation.is_permutation_only() {
        return None;
    }

    // Must have same number of indices
    if input.explicit_count() != output.explicit_count() {
        return None;
    }

    // Build permutation
    let input_indices: Vec<char> = input.named_indices().collect();
    let output_indices: Vec<char> = output.named_indices().collect();

    // Each output index maps to an input index position
    let permutation: Vec<usize> = output_indices
        .iter()
        .filter_map(|c| input_indices.iter().position(|x| x == c))
        .collect();

    if permutation.len() != input_indices.len() {
        return None;
    }

    // Check if it's actually a permutation (not identity)
    let is_identity = permutation.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return None;  // Not actually a transpose
    }

    Some(permutation)
}

/// Checks if the notation represents a reduction.
///
/// Returns the axes to reduce over and the implied operation (sum).
pub fn is_reduction(notation: &EinsumNotation) -> Option<(Vec<usize>, ReductionType)> {
    if !notation.is_unary() {
        return None;
    }

    // Must have contractions (indices in input not in output)
    if notation.is_permutation_only() {
        return None;
    }

    let input = &notation.inputs()[0];
    let output = notation.output();

    let input_indices: Vec<char> = input.named_indices().collect();
    let output_indices: BTreeSet<char> = output.named_indices().collect();

    // Find reduced axes
    let reduced_axes: Vec<usize> = input_indices
        .iter()
        .enumerate()
        .filter(|(_, c)| !output_indices.contains(c))
        .map(|(i, _)| i)
        .collect();

    if reduced_axes.is_empty() {
        return None;
    }

    // Default to sum reduction (einsum semantics)
    Some((reduced_axes, ReductionType::Sum))
}

/// Type of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionType {
    Sum,
    Prod,
    Max,
    Min,
}

/// Checks if the notation represents a trace operation.
///
/// `ii->` computes the trace (sum of diagonal).
pub fn is_trace(notation: &EinsumNotation) -> bool {
    if !notation.is_unary() {
        return false;
    }

    let input = &notation.inputs()[0];
    let output = notation.output();

    // Output must be scalar or have fewer dims
    if !notation.is_scalar_output() {
        return false;
    }

    // Input must have repeated indices (diagonal access)
    let input_indices: Vec<char> = input.named_indices().collect();

    // Check for repeated indices
    let unique: BTreeSet<char> = input_indices.iter().copied().collect();

    // If unique count < total count, we have repeats
    if unique.len() >= input_indices.len() {
        return false;
    }

    // All repeated indices must be contracted (not in output)
    for c in &input_indices {
        let count = input_indices.iter().filter(|x| *x == c).count();
        if count > 1 && output.contains(*c) {
            return false;  // Repeated index in output = not trace
        }
    }

    true
}

/// Checks if the notation represents diagonal extraction.
///
/// `ii->i` extracts the diagonal.
/// `bii->bi` extracts diagonal with batch dim.
pub fn is_diagonal_extract(notation: &EinsumNotation) -> Option<DiagonalConfig> {
    if !notation.is_unary() {
        return None;
    }

    let input = &notation.inputs()[0];
    let output = notation.output();

    let input_indices: Vec<char> = input.named_indices().collect();
    let _output_indices: Vec<char> = output.named_indices().collect();

    // Find repeated indices in input
    let mut seen: BTreeSet<char> = BTreeSet::new();
    let mut repeated: Vec<char> = Vec::new();

    for &c in &input_indices {
        if seen.contains(&c) {
            if !repeated.contains(&c) {
                repeated.push(c);
            }
        } else {
            seen.insert(c);
        }
    }

    if repeated.is_empty() {
        return None;
    }

    // Repeated indices must appear in output (extraction, not trace)
    for &c in &repeated {
        if !output.contains(c) {
            return None;  // This would be trace, not extraction
        }
    }

    // Non-repeated indices in input must all appear in output
    let non_repeated: Vec<char> = input_indices
        .iter()
        .filter(|c| !repeated.contains(c))
        .copied()
        .collect();

    for &c in &non_repeated {
        if !output.contains(c) {
            return None;
        }
    }

    // Find the diagonal dimension
    let diag_dim = input_indices.iter().position(|&c| repeated.contains(&c)).unwrap();

    Some(DiagonalConfig {
        diagonal_index: repeated[0],
        diagonal_dim: diag_dim,
    })
}

/// Configuration for diagonal extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagonalConfig {
    /// The repeated index character.
    pub diagonal_index: char,
    /// Position of the diagonal dimension.
    pub diagonal_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_transpose_2d() {
        let notation = parse_einsum("ij->ji").unwrap();
        let perm = is_transpose(&notation).unwrap();

        assert_eq!(perm, vec![1, 0]);
    }

    #[test]
    fn test_transpose_3d() {
        let notation = parse_einsum("ijk->kji").unwrap();
        let perm = is_transpose(&notation).unwrap();

        assert_eq!(perm, vec![2, 1, 0]);
    }

    #[test]
    fn test_transpose_partial() {
        let notation = parse_einsum("ijkl->jilk").unwrap();
        let perm = is_transpose(&notation).unwrap();

        assert_eq!(perm, vec![1, 0, 3, 2]);
    }

    #[test]
    fn test_not_transpose_identity() {
        let notation = parse_einsum("ij->ij").unwrap();
        assert!(is_transpose(&notation).is_none());
    }

    #[test]
    fn test_reduction_single_axis() {
        let notation = parse_einsum("ij->i").unwrap();
        let (axes, _) = is_reduction(&notation).unwrap();

        assert_eq!(axes, vec![1]);
    }

    #[test]
    fn test_reduction_multiple_axes() {
        let notation = parse_einsum("ijk->j").unwrap();
        let (axes, _) = is_reduction(&notation).unwrap();

        assert_eq!(axes, vec![0, 2]);
    }

    #[test]
    fn test_reduction_all() {
        let notation = parse_einsum("ijk->").unwrap();
        let (axes, _) = is_reduction(&notation).unwrap();

        assert_eq!(axes, vec![0, 1, 2]);
    }

    #[test]
    fn test_trace() {
        let notation = parse_einsum("ii->").unwrap();
        assert!(is_trace(&notation));
    }

    #[test]
    fn test_not_trace_with_output() {
        // iji->j is NOT a trace because output is not scalar
        // It's a reduction over repeated indices, but trace specifically means scalar output
        let notation = parse_einsum("iji->j").unwrap();
        assert!(!is_trace(&notation));
    }

    #[test]
    fn test_trace_3d() {
        // ijj-> is a trace (contracts j, produces scalar)
        let notation = parse_einsum("ijj->").unwrap();
        assert!(is_trace(&notation));
    }

    #[test]
    fn test_diagonal_extract() {
        let notation = parse_einsum("ii->i").unwrap();
        let config = is_diagonal_extract(&notation).unwrap();

        assert_eq!(config.diagonal_index, 'i');
    }

    #[test]
    fn test_diagonal_extract_batched() {
        let notation = parse_einsum("bii->bi").unwrap();
        let config = is_diagonal_extract(&notation).unwrap();

        assert_eq!(config.diagonal_index, 'i');
    }
}
