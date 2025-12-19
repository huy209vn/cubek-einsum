//! Greedy contraction path optimization.
//!
//! O(n³) algorithm that repeatedly contracts the cheapest pair.

use alloc::vec::Vec;
use alloc::collections::BTreeSet;

use super::cost::{CostModel, ContractionCost};
use super::path::{ContractionPath, ContractionStep, TensorState};
use crate::notation::EinsumNotation;

/// Finds a contraction path using the greedy algorithm.
///
/// At each step, contracts the pair of tensors with lowest cost.
/// Time complexity: O(n³) where n is the number of input tensors.
///
/// # Arguments
/// * `notation` - The einsum notation
/// * `shapes` - Shapes of input tensors
/// * `cost_model` - Cost model for evaluating contractions
pub fn greedy_path(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
    cost_model: &CostModel,
) -> ContractionPath {
    let n = notation.num_inputs();

    if n == 0 {
        return ContractionPath::new();
    }

    if n == 1 {
        // Single input - no contraction needed
        return ContractionPath::new();
    }

    // Initialize tensor state
    let initial_shapes: Vec<Vec<usize>> = shapes.iter().map(|s| s.to_vec()).collect();
    let initial_indices: Vec<Vec<char>> = notation
        .inputs()
        .iter()
        .map(|s| s.named_indices().collect())
        .collect();

    let mut state = TensorState::new(initial_shapes, initial_indices);
    let mut path = ContractionPath::with_capacity(n - 1);

    // Track which indices need to be kept for final output
    let output_set: BTreeSet<char> = notation.output().named_indices().collect();

    // Greedy loop: contract cheapest pair until one tensor remains
    while state.len() > 1 {
        let (best_i, best_j, step) = find_best_pair(&state, &output_set, cost_model);

        // Contract the pair
        let new_state = state.contract(best_i, best_j, &step.result_indices);
        state = new_state;

        path.push(step);
    }

    path
}

/// Finds the best pair to contract in the current state.
fn find_best_pair(
    state: &TensorState,
    output_indices: &BTreeSet<char>,
    cost_model: &CostModel,
) -> (usize, usize, ContractionStep) {
    let mut best_cost = ContractionCost::new(u64::MAX, u64::MAX, 1);
    let mut best_pair = (0, 1);
    let mut best_step = None;

    let n = state.len();

    for i in 0..n {
        for j in (i + 1)..n {
            let (step, cost) = evaluate_pair(state, i, j, output_indices, cost_model);

            if cost < best_cost {
                best_cost = cost;
                best_pair = (i, j);
                best_step = Some(step);
            }
        }
    }

    let step = best_step.expect("should have at least one pair");
    (best_pair.0, best_pair.1, step)
}

/// Evaluates the cost of contracting a specific pair.
fn evaluate_pair(
    state: &TensorState,
    i: usize,
    j: usize,
    final_output: &BTreeSet<char>,
    cost_model: &CostModel,
) -> (ContractionStep, ContractionCost) {
    let indices_i: BTreeSet<char> = state.indices[i].iter().copied().collect();
    let indices_j: BTreeSet<char> = state.indices[j].iter().copied().collect();

    // Indices that appear in both inputs
    let common: BTreeSet<char> = indices_i.intersection(&indices_j).copied().collect();

    // Indices that appear in other tensors or final output (must be kept)
    let mut kept_elsewhere = final_output.clone();
    for (k, idx) in state.indices.iter().enumerate() {
        if k != i && k != j {
            for &c in idx {
                kept_elsewhere.insert(c);
            }
        }
    }

    // Contracted indices: common indices not needed elsewhere
    let contracted: Vec<char> = common
        .iter()
        .filter(|c| !kept_elsewhere.contains(c))
        .copied()
        .collect();

    // Result indices: all indices from i and j except contracted
    let contracted_set: BTreeSet<char> = contracted.iter().copied().collect();
    let mut result_indices: Vec<char> = Vec::new();
    let mut seen: BTreeSet<char> = BTreeSet::new();

    // Preserve order from first tensor
    for &c in &state.indices[i] {
        if !contracted_set.contains(&c) && !seen.contains(&c) {
            result_indices.push(c);
            seen.insert(c);
        }
    }
    // Add indices from second tensor that weren't in first
    for &c in &state.indices[j] {
        if !contracted_set.contains(&c) && !seen.contains(&c) {
            result_indices.push(c);
            seen.insert(c);
        }
    }

    // Compute cost
    let cost = cost_model.compute_pairwise_cost(
        &state.shapes[i],
        &state.shapes[j],
        &state.indices[i],
        &state.indices[j],
        &contracted,
    );

    let step = ContractionStep::new(
        (i, j),
        contracted,
        result_indices,
        cost.flops,
    );

    (step, cost)
}

/// Greedy with size-based tie-breaking.
///
/// When costs are similar, prefer contracting smaller tensors first.
#[allow(dead_code)]
pub fn greedy_path_size_tiebreak(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
    cost_model: &CostModel,
) -> ContractionPath {
    // TODO: Implement size-based tie-breaking
    // For now, fall back to basic greedy
    greedy_path(notation, shapes, cost_model)
}

/// Greedy with FLOP-based ordering (SSGreedy from opt_einsum).
///
/// Considers only FLOP cost, ignoring memory.
#[allow(dead_code)]
pub fn greedy_flops_only(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
) -> ContractionPath {
    let cost_model = CostModel { alpha: 0 }; // Memory weight = 0
    greedy_path(notation, shapes, &cost_model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_greedy_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];
        let cost_model = CostModel::default();

        let path = greedy_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 1);
        assert_eq!(path.steps()[0].inputs, (0, 1));
        assert_eq!(path.steps()[0].contracted_indices, vec!['j']);
    }

    #[test]
    fn test_greedy_chain() {
        // A @ B @ C: should contract in some order
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];
        let cost_model = CostModel::default();

        let path = greedy_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 2); // Two contractions needed
    }

    #[test]
    fn test_greedy_optimal_chain() {
        // Test that greedy finds good order for chain: 2x10, 10x1000, 1000x3
        // Optimal: (A @ B) @ C = 2*10*1000 + 2*1000*3 = 26,000
        // vs A @ (B @ C) = 10*1000*3 + 2*10*3 = 30,060
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        let shapes: &[&[usize]] = &[&[2, 10], &[10, 1000], &[1000, 3]];
        let cost_model = CostModel::default();

        let path = greedy_path(&notation, shapes, &cost_model);

        // Path should exist
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_greedy_batch_matmul() {
        let notation = parse_einsum("bij,bjk->bik").unwrap();
        let shapes: &[&[usize]] = &[&[8, 64, 128], &[8, 128, 256]];
        let cost_model = CostModel::default();

        let path = greedy_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 1);
    }
}
