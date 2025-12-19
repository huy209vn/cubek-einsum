//! Binary operation pattern detection.

use alloc::collections::BTreeSet;

use crate::notation::EinsumNotation;

/// Checks if the notation represents a Hadamard (element-wise) product.
///
/// `ij,ij->ij` - same indices in both inputs and output.
pub fn is_hadamard(notation: &EinsumNotation) -> bool {
    if !notation.is_binary() {
        return false;
    }

    // Must have no contractions
    if !notation.is_permutation_only() {
        return false;
    }

    let inputs = notation.inputs();
    let output = notation.output();

    let indices_a: BTreeSet<char> = inputs[0].named_indices().collect();
    let indices_b: BTreeSet<char> = inputs[1].named_indices().collect();
    let indices_out: BTreeSet<char> = output.named_indices().collect();

    // All three must have the same indices
    indices_a == indices_b && indices_a == indices_out
}

/// Checks if the notation represents an outer product.
///
/// `i,j->ij` - disjoint indices, output is their concatenation.
pub fn is_outer_product(notation: &EinsumNotation) -> bool {
    if !notation.is_binary() {
        return false;
    }

    // Must have no contractions
    if !notation.is_permutation_only() {
        return false;
    }

    let inputs = notation.inputs();
    let output = notation.output();

    let indices_a: BTreeSet<char> = inputs[0].named_indices().collect();
    let indices_b: BTreeSet<char> = inputs[1].named_indices().collect();
    let indices_out: BTreeSet<char> = output.named_indices().collect();

    // Inputs must be disjoint
    if !indices_a.is_disjoint(&indices_b) {
        return false;
    }

    // Output must be union of inputs
    let union: BTreeSet<char> = indices_a.union(&indices_b).copied().collect();
    union == indices_out
}

/// Checks if the notation represents a dot product.
///
/// `i,i->` - same indices in both inputs, scalar output.
/// Also matches `ij,ij->` (Frobenius inner product).
pub fn is_dot_product(notation: &EinsumNotation) -> bool {
    if !notation.is_binary() {
        return false;
    }

    // Output must be scalar
    if !notation.is_scalar_output() {
        return false;
    }

    let inputs = notation.inputs();

    let indices_a: BTreeSet<char> = inputs[0].named_indices().collect();
    let indices_b: BTreeSet<char> = inputs[1].named_indices().collect();

    // Both inputs must have the same indices
    if indices_a != indices_b {
        return false;
    }

    // All indices must be contracted (since output is scalar)
    true
}

/// Checks if the notation represents a bilinear form.
///
/// `i,ij,j->` - vector-matrix-vector product.
#[allow(dead_code)]
pub fn is_bilinear_form(notation: &EinsumNotation) -> bool {
    if notation.num_inputs() != 3 {
        return false;
    }

    // TODO: Implement bilinear form detection
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_hadamard_2d() {
        let notation = parse_einsum("ij,ij->ij").unwrap();
        assert!(is_hadamard(&notation));
    }

    #[test]
    fn test_hadamard_3d() {
        let notation = parse_einsum("ijk,ijk->ijk").unwrap();
        assert!(is_hadamard(&notation));
    }

    #[test]
    fn test_not_hadamard_different_indices() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        assert!(!is_hadamard(&notation));
    }

    #[test]
    fn test_outer_product_1d() {
        let notation = parse_einsum("i,j->ij").unwrap();
        assert!(is_outer_product(&notation));
    }

    #[test]
    fn test_outer_product_2d() {
        let notation = parse_einsum("ij,kl->ijkl").unwrap();
        assert!(is_outer_product(&notation));
    }

    #[test]
    fn test_not_outer_shared_index() {
        let notation = parse_einsum("ij,jk->ijk").unwrap();
        assert!(!is_outer_product(&notation));
    }

    #[test]
    fn test_dot_product_1d() {
        let notation = parse_einsum("i,i->").unwrap();
        assert!(is_dot_product(&notation));
    }

    #[test]
    fn test_frobenius_inner_product() {
        let notation = parse_einsum("ij,ij->").unwrap();
        assert!(is_dot_product(&notation));
    }

    #[test]
    fn test_not_dot_product_with_output() {
        let notation = parse_einsum("i,i->i").unwrap();
        assert!(!is_dot_product(&notation));
    }
}
