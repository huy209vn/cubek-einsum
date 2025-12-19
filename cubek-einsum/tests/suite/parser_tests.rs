//! Parser tests for einsum notation.

use cubek_einsum::notation::parse_einsum;

#[test]
fn test_parse_basic_matmul() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    assert_eq!(notation.num_inputs(), 2);
    assert!(!notation.has_ellipsis());
}

#[test]
fn test_parse_batched_matmul() {
    let notation = parse_einsum("bij,bjk->bik").unwrap();
    assert_eq!(notation.num_inputs(), 2);
    assert!(notation.batch_indices().contains(&'b'));
}

#[test]
fn test_parse_attention() {
    let notation = parse_einsum("bhqd,bhkd->bhqk").unwrap();
    assert_eq!(notation.num_inputs(), 2);
    assert!(notation.contraction_indices().contains(&'d'));
}

#[test]
fn test_parse_transpose() {
    let notation = parse_einsum("ij->ji").unwrap();
    assert!(notation.is_unary());
    assert!(notation.is_permutation_only());
}

#[test]
fn test_parse_trace() {
    let notation = parse_einsum("ii->").unwrap();
    assert!(notation.is_unary());
    assert!(notation.is_scalar_output());
}

#[test]
fn test_parse_ellipsis() {
    let notation = parse_einsum("...ij,...jk->...ik").unwrap();
    assert!(notation.has_ellipsis());
}

#[test]
fn test_parse_implicit_output() {
    let notation = parse_einsum("ij,jk").unwrap();
    // Should imply ->ik
    let output_str = notation.output().to_string();
    assert!(output_str.contains('i'));
    assert!(output_str.contains('k'));
    assert!(!output_str.contains('j'));
}

#[test]
fn test_parse_chain() {
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    assert_eq!(notation.num_inputs(), 3);
}
