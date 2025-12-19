//! Tests for chain contractions (multi-step operations).

use cubek_einsum::notation::{parse_einsum, EinsumNotation};
use cubek_einsum::optimization::{create_plan, ContractionStrategy};

#[test]
fn test_chain_contraction_plan() {
    // Test: ij,jk,kl->il (chain of 3 tensors)
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

    // Should not use fast path (too complex)
    assert!(!plan.uses_fast_path());
    
    // Should have 2 contraction steps: (ij,jk)->ik, then (ik,kl)->il
    assert_eq!(plan.num_steps(), 2);
}

#[test]
fn test_chain_with_reduction_plan() {
    // Test: ij,jk->i (matmul followed by reduction)
    let notation = parse_einsum("ij,jk->i").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

    // This should use fast path (matmul with reduction)
    assert!(plan.uses_fast_path());
}

#[test]
fn test_simple_chain_plan() {
    // Test: ij,jk->ik (simple matmul - should use fast path)
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

    // Should use fast path (matmul)
    assert!(plan.uses_fast_path());
}

#[test]
fn test_two_step_chain_plan() {
    // Test: ij,jk->ik (should be 1 step, not a chain)
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

    // Should use fast path (matmul)
    assert!(plan.uses_fast_path());
}

#[test]
fn test_three_tensor_chain_plan() {
    // Test: ij,jk,kl->il (chain of 3 tensors)
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

    // Should not use fast path (too complex)
    assert!(!plan.uses_fast_path());
    
    // Should have exactly 2 steps for this chain
    assert_eq!(plan.num_steps(), 2);
}

#[test]
fn test_four_tensor_chain_plan() {
    // Test: ij,jk,kl,lm->im (chain of 4 tensors)
    let notation = parse_einsum("ij,jk,kl,lm->im").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40], &[40, 50]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

    // Should not use fast path (too complex)
    assert!(!plan.uses_fast_path());
    
    // Should have exactly 3 steps for this chain
    assert_eq!(plan.num_steps(), 3);
}
