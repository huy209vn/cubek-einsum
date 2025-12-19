//! Branch and bound contraction path optimization.
//!
//! Finds optimal or near-optimal contraction paths by exploring the search space
//! with aggressive pruning. Good balance between greedy (fast, suboptimal) and
//! full DP (optimal, exponential).

use alloc::vec::Vec;
use alloc::collections::BTreeSet;

use super::cost::{CostModel, ContractionCost};
use super::path::{ContractionPath, ContractionStep, TensorState};
use super::greedy::greedy_path;
use crate::notation::EinsumNotation;

/// Maximum depth before falling back to greedy.
const MAX_SEARCH_DEPTH: usize = 8;

/// Maximum nodes to explore before returning best found.
const MAX_NODES: u64 = 100_000;

/// Branch and bound search state.
struct SearchState<'a> {
    #[allow(dead_code)]
    notation: &'a EinsumNotation,
    cost_model: &'a CostModel,
    output_indices: BTreeSet<char>,

    /// Best path found so far.
    best_path: Option<ContractionPath>,
    /// Cost of best path.
    best_cost: ContractionCost,
    /// Number of nodes explored.
    nodes_explored: u64,
}

/// Finds a contraction path using branch and bound.
///
/// Explores the search space of contraction orderings with pruning:
/// 1. Upper bound: best solution found so far (starts with greedy)
/// 2. Lower bound: optimistic estimate of remaining cost
/// 3. Prunes branches where lower_bound >= best_cost
///
/// # Arguments
/// * `notation` - The einsum notation
/// * `shapes` - Shapes of input tensors
/// * `cost_model` - Cost model for evaluating contractions
pub fn branch_bound_path(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
    cost_model: &CostModel,
) -> ContractionPath {
    let n = notation.num_inputs();

    if n == 0 || n == 1 {
        return ContractionPath::new();
    }

    // Initialize tensor state
    let initial_shapes: Vec<Vec<usize>> = shapes.iter().map(|s| s.to_vec()).collect();
    let initial_indices: Vec<Vec<char>> = notation
        .inputs()
        .iter()
        .map(|s| s.named_indices().collect())
        .collect();

    let output_indices: BTreeSet<char> = notation.output().named_indices().collect();

    // Get greedy solution as initial upper bound
    let greedy = greedy_path(notation, shapes, cost_model);
    let greedy_cost = compute_path_cost(&greedy, &initial_shapes, &initial_indices, cost_model);

    let mut state = SearchState {
        notation,
        cost_model,
        output_indices,
        best_path: Some(greedy),
        best_cost: greedy_cost,
        nodes_explored: 0,
    };

    // Start search
    let tensor_state = TensorState::new(initial_shapes, initial_indices);
    let mut current_path = Vec::new();
    let current_cost = ContractionCost::zero();

    branch_bound_search(
        &mut state,
        tensor_state,
        &mut current_path,
        current_cost,
        0,
    );

    state.best_path.unwrap_or_else(ContractionPath::new)
}

/// Recursive branch and bound search.
fn branch_bound_search(
    state: &mut SearchState,
    tensor_state: TensorState,
    current_path: &mut Vec<ContractionStep>,
    current_cost: ContractionCost,
    depth: usize,
) {
    state.nodes_explored += 1;

    // Check termination conditions
    if state.nodes_explored >= MAX_NODES {
        return;
    }

    // Base case: only one tensor left
    if tensor_state.len() <= 1 {
        if current_cost < state.best_cost {
            state.best_cost = current_cost;
            let mut path = ContractionPath::with_capacity(current_path.len());
            for step in current_path.iter() {
                path.push(step.clone());
            }
            state.best_path = Some(path);
        }
        return;
    }

    // If too deep, use greedy for remainder
    if depth >= MAX_SEARCH_DEPTH {
        let remaining_cost = greedy_remaining_cost(
            &tensor_state,
            &state.output_indices,
            state.cost_model,
        );
        let total_cost = current_cost + remaining_cost;

        if total_cost < state.best_cost {
            // Reconstruct full path with greedy remainder
            let greedy_steps = greedy_remaining_steps(
                &tensor_state,
                &state.output_indices,
                state.cost_model,
            );

            state.best_cost = total_cost;
            let mut path = ContractionPath::with_capacity(current_path.len() + greedy_steps.len());
            for step in current_path.iter() {
                path.push(step.clone());
            }
            for step in greedy_steps {
                path.push(step);
            }
            state.best_path = Some(path);
        }
        return;
    }

    // Compute lower bound on remaining cost
    let lower_bound = state.cost_model.optimistic_remaining_cost(
        &tensor_state.shapes,
        &tensor_state.indices,
    );

    // Prune if we can't possibly beat the best
    if current_cost + lower_bound >= state.best_cost {
        return;
    }

    // Generate all possible contractions, sorted by cost (best first)
    let mut candidates = generate_candidates(&tensor_state, &state.output_indices, state.cost_model);

    // Sort by cost (greedy ordering for better pruning)
    candidates.sort_by(|a, b| a.2.cmp(&b.2));

    // Try each candidate
    for (i, j, step_cost, step) in candidates {
        // Check if this branch can beat current best
        let new_cost = current_cost + step_cost;

        if new_cost >= state.best_cost {
            // Since candidates are sorted, remaining candidates are worse
            break;
        }

        // Apply contraction
        let new_tensor_state = tensor_state.contract(i, j, &step.result_indices);

        // Recurse
        current_path.push(step);
        branch_bound_search(state, new_tensor_state, current_path, new_cost, depth + 1);
        current_path.pop();

        // Early termination if we've hit node limit
        if state.nodes_explored >= MAX_NODES {
            return;
        }
    }
}

/// Generates all possible pairwise contractions for the current state.
fn generate_candidates(
    state: &TensorState,
    output_indices: &BTreeSet<char>,
    cost_model: &CostModel,
) -> Vec<(usize, usize, ContractionCost, ContractionStep)> {
    let n = state.len();
    let mut candidates = Vec::with_capacity(n * (n - 1) / 2);

    // Track which indices are needed elsewhere
    let mut kept_indices = output_indices.clone();
    for idx_vec in &state.indices {
        for &c in idx_vec {
            kept_indices.insert(c);
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let indices_i: BTreeSet<char> = state.indices[i].iter().copied().collect();
            let indices_j: BTreeSet<char> = state.indices[j].iter().copied().collect();

            // Common indices
            let common: BTreeSet<char> = indices_i.intersection(&indices_j).copied().collect();

            // Indices that must be kept (appear in other tensors or output)
            let mut must_keep = output_indices.clone();
            for (k, idx) in state.indices.iter().enumerate() {
                if k != i && k != j {
                    for &c in idx {
                        must_keep.insert(c);
                    }
                }
            }

            // Contracted indices
            let contracted: Vec<char> = common
                .iter()
                .filter(|c| !must_keep.contains(c))
                .copied()
                .collect();

            let contracted_set: BTreeSet<char> = contracted.iter().copied().collect();

            // Result indices
            let mut result_indices = Vec::new();
            let mut seen = BTreeSet::new();

            for &c in &state.indices[i] {
                if !contracted_set.contains(&c) && !seen.contains(&c) {
                    result_indices.push(c);
                    seen.insert(c);
                }
            }
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

            candidates.push((i, j, cost, step));
        }
    }

    candidates
}

/// Computes the total cost of a path.
fn compute_path_cost(
    path: &ContractionPath,
    initial_shapes: &[Vec<usize>],
    initial_indices: &[Vec<char>],
    cost_model: &CostModel,
) -> ContractionCost {
    let mut total = ContractionCost::zero();
    let mut state = TensorState::new(initial_shapes.to_vec(), initial_indices.to_vec());

    for step in path.steps() {
        let (i, j) = step.inputs;

        let cost = cost_model.compute_pairwise_cost(
            &state.shapes[i],
            &state.shapes[j],
            &state.indices[i],
            &state.indices[j],
            &step.contracted_indices,
        );

        total = total + cost;
        state = state.contract(i, j, &step.result_indices);
    }

    total
}

/// Computes cost of greedy solution for remaining tensors.
fn greedy_remaining_cost(
    state: &TensorState,
    output_indices: &BTreeSet<char>,
    cost_model: &CostModel,
) -> ContractionCost {
    let steps = greedy_remaining_steps(state, output_indices, cost_model);

    let mut total = ContractionCost::zero();
    let mut current_state = state.clone();

    for step in &steps {
        let (i, j) = step.inputs;

        let cost = cost_model.compute_pairwise_cost(
            &current_state.shapes[i],
            &current_state.shapes[j],
            &current_state.indices[i],
            &current_state.indices[j],
            &step.contracted_indices,
        );

        total = total + cost;
        current_state = current_state.contract(i, j, &step.result_indices);
    }

    total
}

/// Gets greedy steps for remaining tensors.
fn greedy_remaining_steps(
    state: &TensorState,
    output_indices: &BTreeSet<char>,
    cost_model: &CostModel,
) -> Vec<ContractionStep> {
    let mut steps = Vec::new();
    let mut current_state = state.clone();

    while current_state.len() > 1 {
        let candidates = generate_candidates(&current_state, output_indices, cost_model);

        // Pick lowest cost
        let best = candidates
            .into_iter()
            .min_by(|a, b| a.2.cmp(&b.2))
            .expect("should have candidates");

        let (i, j, _, step) = best;
        current_state = current_state.contract(i, j, &step.result_indices);
        steps.push(step);
    }

    steps
}

/// Branch and bound with configurable limits.
#[allow(dead_code)]
pub fn branch_bound_path_with_limits(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
    cost_model: &CostModel,
    max_nodes: u64,
    max_depth: usize,
) -> ContractionPath {
    let n = notation.num_inputs();

    if n == 0 || n == 1 {
        return ContractionPath::new();
    }

    let initial_shapes: Vec<Vec<usize>> = shapes.iter().map(|s| s.to_vec()).collect();
    let initial_indices: Vec<Vec<char>> = notation
        .inputs()
        .iter()
        .map(|s| s.named_indices().collect())
        .collect();

    let output_indices: BTreeSet<char> = notation.output().named_indices().collect();

    let greedy = greedy_path(notation, shapes, cost_model);
    let greedy_cost = compute_path_cost(&greedy, &initial_shapes, &initial_indices, cost_model);

    let mut search_state = SearchState {
        notation,
        cost_model,
        output_indices: output_indices.clone(),
        best_path: Some(greedy),
        best_cost: greedy_cost,
        nodes_explored: 0,
    };

    // Custom search with limits
    let tensor_state = TensorState::new(initial_shapes, initial_indices);
    let mut current_path = Vec::new();
    let current_cost = ContractionCost::zero();

    branch_bound_search_with_limits(
        &mut search_state,
        tensor_state,
        &mut current_path,
        current_cost,
        0,
        max_nodes,
        max_depth,
        &output_indices,
    );

    search_state.best_path.unwrap_or_else(ContractionPath::new)
}

fn branch_bound_search_with_limits(
    state: &mut SearchState,
    tensor_state: TensorState,
    current_path: &mut Vec<ContractionStep>,
    current_cost: ContractionCost,
    depth: usize,
    max_nodes: u64,
    max_depth: usize,
    output_indices: &BTreeSet<char>,
) {
    state.nodes_explored += 1;

    if state.nodes_explored >= max_nodes {
        return;
    }

    if tensor_state.len() <= 1 {
        if current_cost < state.best_cost {
            state.best_cost = current_cost;
            let mut path = ContractionPath::with_capacity(current_path.len());
            for step in current_path.iter() {
                path.push(step.clone());
            }
            state.best_path = Some(path);
        }
        return;
    }

    if depth >= max_depth {
        let remaining_cost = greedy_remaining_cost(
            &tensor_state,
            output_indices,
            state.cost_model,
        );
        let total_cost = current_cost + remaining_cost;

        if total_cost < state.best_cost {
            let greedy_steps = greedy_remaining_steps(
                &tensor_state,
                output_indices,
                state.cost_model,
            );

            state.best_cost = total_cost;
            let mut path = ContractionPath::with_capacity(current_path.len() + greedy_steps.len());
            for step in current_path.iter() {
                path.push(step.clone());
            }
            for step in greedy_steps {
                path.push(step);
            }
            state.best_path = Some(path);
        }
        return;
    }

    let lower_bound = state.cost_model.optimistic_remaining_cost(
        &tensor_state.shapes,
        &tensor_state.indices,
    );

    if current_cost + lower_bound >= state.best_cost {
        return;
    }

    let mut candidates = generate_candidates(&tensor_state, output_indices, state.cost_model);
    candidates.sort_by(|a, b| a.2.cmp(&b.2));

    for (i, j, step_cost, step) in candidates {
        let new_cost = current_cost + step_cost;

        if new_cost >= state.best_cost {
            break;
        }

        let new_tensor_state = tensor_state.contract(i, j, &step.result_indices);

        current_path.push(step);
        branch_bound_search_with_limits(
            state,
            new_tensor_state,
            current_path,
            new_cost,
            depth + 1,
            max_nodes,
            max_depth,
            output_indices,
        );
        current_path.pop();

        if state.nodes_explored >= max_nodes {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_branch_bound_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];
        let cost_model = CostModel::default();

        let path = branch_bound_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 1);
        assert_eq!(path.steps()[0].inputs, (0, 1));
    }

    #[test]
    fn test_branch_bound_chain() {
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];
        let cost_model = CostModel::default();

        let path = branch_bound_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_branch_bound_vs_greedy() {
        // Skewed chain where order matters
        // 2x10, 10x1000, 1000x3
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        let shapes: &[&[usize]] = &[&[2, 10], &[10, 1000], &[1000, 3]];
        let cost_model = CostModel { alpha: 0 };

        let bb_path = branch_bound_path(&notation, shapes, &cost_model);
        let greedy = greedy_path(&notation, shapes, &cost_model);

        // Both should find a valid path
        assert_eq!(bb_path.len(), 2);
        assert_eq!(greedy.len(), 2);

        // B&B should be at least as good as greedy
        // (may be equal if greedy found optimal)
    }

    #[test]
    fn test_branch_bound_four_tensors() {
        let notation = parse_einsum("ij,jk,kl,lm->im").unwrap();
        let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40], &[40, 50]];
        let cost_model = CostModel::default();

        let path = branch_bound_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_branch_bound_batch_matmul() {
        let notation = parse_einsum("bij,bjk->bik").unwrap();
        let shapes: &[&[usize]] = &[&[8, 64, 128], &[8, 128, 256]];
        let cost_model = CostModel::default();

        let path = branch_bound_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_branch_bound_with_limits() {
        let notation = parse_einsum("ij,jk,kl,lm,mn->in").unwrap();
        let shapes: &[&[usize]] = &[
            &[10, 20], &[20, 30], &[30, 40], &[40, 50], &[50, 60]
        ];
        let cost_model = CostModel::default();

        // With low limits, should still produce valid path
        let path = branch_bound_path_with_limits(
            &notation, shapes, &cost_model,
            100,  // max_nodes
            3,    // max_depth
        );

        assert_eq!(path.len(), 4);
    }
}
