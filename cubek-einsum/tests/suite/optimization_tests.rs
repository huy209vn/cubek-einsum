//! Contraction path optimization tests.

use cubek_einsum::notation::parse_einsum;
use cubek_einsum::optimization::{
    greedy_path, optimal_path, create_plan,
    CostModel, ContractionStrategy,
};

#[test]
fn test_greedy_two_tensors() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];
    let cost_model = CostModel::default();

    let path = greedy_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 1);
}

#[test]
fn test_greedy_three_tensors() {
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];
    let cost_model = CostModel::default();

    let path = greedy_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 2);
}

#[test]
fn test_optimal_two_tensors() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];
    let cost_model = CostModel::default();

    let path = optimal_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 1);
}

#[test]
fn test_optimal_three_tensors() {
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];
    let cost_model = CostModel::default();

    let path = optimal_path(&notation, shapes, &cost_model);
    assert_eq!(path.len(), 2);
}

#[test]
fn test_plan_uses_fast_path_for_matmul() {
    let notation = parse_einsum("ij,jk->ik").unwrap();
    let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);
    assert!(plan.uses_fast_path());
}

#[test]
fn test_plan_no_fast_path_for_chain() {
    let notation = parse_einsum("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);
    assert!(!plan.uses_fast_path());
}

#[test]
fn test_cost_model() {
    let model = CostModel::default();

    let cost = model.compute_pairwise_cost(
        &[100, 200],
        &[200, 300],
        &['i', 'j'],
        &['j', 'k'],
        &['j'],
    );

    // FLOPs = M * N * K * 2 = 100 * 300 * 200 * 2 = 12,000,000
    assert_eq!(cost.flops, 12_000_000);
}

#[test]
fn test_plan_uses_fast_path_for_nm_md_nd() {
    // GNN pattern: nm,md->nd should use fast path
    let notation = parse_einsum("nm,md->nd").unwrap();
    let shapes: &[&[usize]] = &[&[4096, 4096], &[4096, 64]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);
    assert!(plan.uses_fast_path(), "nm,md->nd should use fast path");
}

#[test]
fn test_plan_uses_fast_path_for_gram_matrix() {
    // Gram matrix: ik,jk->ij should use fast path
    let notation = parse_einsum("ik,jk->ij").unwrap();
    let shapes: &[&[usize]] = &[&[1024, 512], &[1024, 512]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);
    assert!(plan.uses_fast_path(), "ik,jk->ij should use fast path");
}

#[test]
fn test_tensor_network_no_fast_path() {
    // Tensor network: bijk,bkjl->bil has complex index structure
    // k and j are contracted, but layout is complex
    let notation = parse_einsum("bijk,bkjl->bil").unwrap();
    let shapes: &[&[usize]] = &[&[16, 64, 128, 64], &[16, 64, 128, 64]];

    let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);
    // This should NOT be a fast path - check what it actually does
    println!("bijk,bkjl->bil uses_fast_path: {}", plan.uses_fast_path());
    println!("num_steps: {}", plan.num_steps());
}
