//! Pattern recognition tests.

use cubek_einsum::notation::parse_einsum;
use cubek_einsum::pattern::{recognize_pattern, FastPath};

#[test]
fn test_recognize_matmul() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::Matmul { .. })));
}

#[test]
fn test_recognize_matmul_transpose_a() {
    let notation = parse_einsum("ji,jk->ik").unwrap();
    let pattern = recognize_pattern(&notation);
    match pattern {
        Some(FastPath::Matmul { transpose_a, transpose_b }) => {
            assert!(transpose_a);
            assert!(!transpose_b);
        }
        _ => panic!("expected matmul pattern"),
    }
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
    match pattern {
        Some(FastPath::Transpose { permutation }) => {
            assert_eq!(permutation, vec![1, 0]);
        }
        _ => panic!("expected transpose pattern"),
    }
}

#[test]
fn test_recognize_reduction() {
    let notation = parse_einsum("ij->i").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::Reduce { .. })));
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
fn test_no_pattern_complex() {
    let notation = parse_einsum("ijk,jkl,klm->im").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(pattern.is_none());
}

#[test]
fn test_recognize_matmul_different_indices() {
    // nm,md->nd should be recognized as matmul (same structure as ij,jk->ik)
    let notation = parse_einsum("nm,md->nd").unwrap();
    let pattern = recognize_pattern(&notation);
    assert!(matches!(pattern, Some(FastPath::Matmul { .. })),
        "nm,md->nd should be Matmul, got {:?}", pattern);
}

#[test]
fn test_recognize_gram_matrix() {
    // ik,jk->ij is a Gram matrix (A @ B^T)
    let notation = parse_einsum("ik,jk->ij").unwrap();
    let pattern = recognize_pattern(&notation);
    match pattern {
        Some(FastPath::Matmul { transpose_a, transpose_b }) => {
            // A is not transposed, B is transposed
            assert!(!transpose_a, "A should not be transposed");
            assert!(transpose_b, "B should be transposed for Gram matrix");
        }
        _ => panic!("ik,jk->ij should be Matmul pattern, got {:?}", pattern),
    }
}

#[test]
fn test_recognize_inner_product_matrix() {
    // ki,kj->ij is A^T @ B
    let notation = parse_einsum("ki,kj->ij").unwrap();
    let pattern = recognize_pattern(&notation);
    match pattern {
        Some(FastPath::Matmul { transpose_a, transpose_b }) => {
            assert!(transpose_a, "A should be transposed");
            assert!(!transpose_b, "B should not be transposed");
        }
        _ => panic!("ki,kj->ij should be Matmul pattern, got {:?}", pattern),
    }
}
