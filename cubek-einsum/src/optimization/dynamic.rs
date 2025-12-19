//! Optimal contraction path via dynamic programming.
//!
//! Finds the globally optimal contraction order by trying all possible
//! bipartitions. Exponential in the number of tensors, but optimal.

use alloc::vec::Vec;
use alloc::collections::BTreeSet;
use hashbrown::HashMap;

use super::cost::{CostModel, ContractionCost};
use super::path::{ContractionPath, ContractionStep};
use crate::notation::EinsumNotation;

/// Maximum number of tensors for which DP is feasible.
/// For n tensors, we have 2^n subsets to consider.
pub const MAX_DP_TENSORS: usize = 12;

/// Finds the optimal contraction path using dynamic programming.
///
/// Uses memoization over all subsets of tensors to find the globally
/// optimal contraction order. Time complexity: O(3^n) where n is
/// the number of input tensors.
///
/// # Arguments
/// * `notation` - The einsum notation
/// * `shapes` - Shapes of input tensors
/// * `cost_model` - Cost model for evaluating contractions
///
/// # Panics
/// Panics if the number of tensors exceeds MAX_DP_TENSORS.
pub fn optimal_path(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
    cost_model: &CostModel,
) -> ContractionPath {
    let n = notation.num_inputs();

    if n == 0 || n == 1 {
        return ContractionPath::new();
    }

    assert!(
        n <= MAX_DP_TENSORS,
        "DP is only feasible for {} tensors or fewer, got {}",
        MAX_DP_TENSORS,
        n
    );

    // Initialize tensor info
    let tensor_shapes: Vec<Vec<usize>> = shapes.iter().map(|s| s.to_vec()).collect();
    let tensor_indices: Vec<Vec<char>> = notation
        .inputs()
        .iter()
        .map(|s| s.named_indices().collect())
        .collect();

    let output_set: BTreeSet<char> = notation.output().named_indices().collect();

    // Memoization table: subset -> (cost, path)
    let mut memo: HashMap<u32, (ContractionCost, Vec<(usize, usize)>)> = HashMap::new();

    // Shape/indices after contracting a subset
    let mut result_cache: HashMap<u32, (Vec<usize>, Vec<char>)> = HashMap::new();

    // Initialize single-tensor subsets
    for i in 0..n {
        let subset = 1u32 << i;
        memo.insert(subset, (ContractionCost::zero(), Vec::new()));
        result_cache.insert(subset, (tensor_shapes[i].clone(), tensor_indices[i].clone()));
    }

    // DP over subset sizes
    for size in 2..=n {
        for subset in subsets_of_size(n, size) {
            let mut best_cost = ContractionCost::new(u64::MAX, u64::MAX, 1);
            let mut best_path = Vec::new();
            let mut best_result: Option<(Vec<usize>, Vec<char>)> = None;

            // Try all bipartitions
            for left in proper_subsets(subset) {
                let right = subset ^ left;
                if right == 0 || left == 0 {
                    continue;
                }
                if left > right {
                    // Avoid duplicate bipartitions
                    continue;
                }

                let (left_cost, left_path) = memo.get(&left).unwrap();
                let (right_cost, right_path) = memo.get(&right).unwrap();

                let (left_shape, left_indices) = result_cache.get(&left).unwrap();
                let (right_shape, right_indices) = result_cache.get(&right).unwrap();

                // Compute contraction cost and result
                let (contract_cost, result_shape, result_indices) = compute_contraction(
                    left_shape,
                    left_indices,
                    right_shape,
                    right_indices,
                    &output_set,
                    subset == (1u32 << n) - 1, // is_final
                    cost_model,
                );

                let total_cost = *left_cost + *right_cost + contract_cost;

                if total_cost < best_cost {
                    best_cost = total_cost;

                    // Build path: left path + right path + this contraction
                    best_path = left_path.clone();
                    best_path.extend(right_path.iter().cloned());

                    // Add this contraction
                    // For path representation, we need original tensor indices
                    let left_tensors: Vec<usize> = (0..n).filter(|&i| (left & (1 << i)) != 0).collect();
                    let right_tensors: Vec<usize> = (0..n).filter(|&i| (right & (1 << i)) != 0).collect();

                    // Use min index from each subset to represent
                    let left_rep = *left_tensors.iter().min().unwrap();
                    let right_rep = *right_tensors.iter().min().unwrap();
                    best_path.push((left_rep, right_rep));

                    best_result = Some((result_shape, result_indices));
                }
            }

            memo.insert(subset, (best_cost, best_path));
            if let Some(result) = best_result {
                result_cache.insert(subset, result);
            }
        }
    }

    // Extract final path
    let full_subset = (1u32 << n) - 1;
    let (_, pair_path) = memo.get(&full_subset).unwrap();

    // Convert pair path to ContractionPath
    build_contraction_path(pair_path, &tensor_indices, &output_set)
}

/// Computes the contraction of two tensor results.
fn compute_contraction(
    shape_a: &[usize],
    indices_a: &[char],
    shape_b: &[usize],
    indices_b: &[char],
    output_set: &BTreeSet<char>,
    is_final: bool,
    cost_model: &CostModel,
) -> (ContractionCost, Vec<usize>, Vec<char>) {
    let indices_a_set: BTreeSet<char> = indices_a.iter().copied().collect();
    let indices_b_set: BTreeSet<char> = indices_b.iter().copied().collect();

    // Common indices
    let common: BTreeSet<char> = indices_a_set.intersection(&indices_b_set).copied().collect();

    // For intermediate results, contract common indices not in final output
    // For final result, contract all common indices not in output
    let contracted: Vec<char> = if is_final {
        common.iter().filter(|c| !output_set.contains(c)).copied().collect()
    } else {
        // For intermediate, we might want to keep some indices
        // Actually, for pairwise contraction, we contract common indices
        common.iter().filter(|c| !output_set.contains(c)).copied().collect()
    };

    let contracted_set: BTreeSet<char> = contracted.iter().copied().collect();

    // Result indices
    let mut result_indices: Vec<char> = Vec::new();
    let mut seen: BTreeSet<char> = BTreeSet::new();

    for &c in indices_a {
        if !contracted_set.contains(&c) && !seen.contains(&c) {
            result_indices.push(c);
            seen.insert(c);
        }
    }
    for &c in indices_b {
        if !contracted_set.contains(&c) && !seen.contains(&c) {
            result_indices.push(c);
            seen.insert(c);
        }
    }

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (&c, &d) in indices_a.iter().zip(shape_a.iter()) {
        dim_map.insert(c, d);
    }
    for (&c, &d) in indices_b.iter().zip(shape_b.iter()) {
        dim_map.insert(c, d);
    }

    // Compute result shape
    let result_shape: Vec<usize> = result_indices
        .iter()
        .map(|c| dim_map.get(c).copied().unwrap_or(1))
        .collect();

    // Compute cost
    let cost = cost_model.compute_pairwise_cost(
        shape_a,
        shape_b,
        indices_a,
        indices_b,
        &contracted,
    );

    (cost, result_shape, result_indices)
}

/// Generates all subsets of {0..n-1} with exactly `size` elements.
fn subsets_of_size(n: usize, size: usize) -> Vec<u32> {
    let mut result = Vec::new();
    generate_subsets(n, size, 0, 0, &mut result);
    result
}

fn generate_subsets(n: usize, size: usize, start: usize, current: u32, result: &mut Vec<u32>) {
    if size == 0 {
        result.push(current);
        return;
    }
    if start >= n {
        return;
    }
    if n - start < size {
        return;
    }

    // Include start
    generate_subsets(n, size - 1, start + 1, current | (1 << start), result);
    // Exclude start
    generate_subsets(n, size, start + 1, current, result);
}

/// Generates all proper non-empty subsets of a set.
fn proper_subsets(set: u32) -> impl Iterator<Item = u32> {
    let mut subset = set;
    core::iter::from_fn(move || {
        if subset == 0 {
            return None;
        }
        let result = subset;
        subset = (subset - 1) & set;
        Some(result)
    })
}

/// Builds a ContractionPath from a list of (left, right) tensor pairs.
fn build_contraction_path(
    pairs: &[(usize, usize)],
    initial_indices: &[Vec<char>],
    output_set: &BTreeSet<char>,
) -> ContractionPath {
    let mut path = ContractionPath::with_capacity(pairs.len());

    // Simulate the contraction process to get correct indices at each step
    let mut current_indices: Vec<Vec<char>> = initial_indices.to_vec();
    let mut index_to_position: Vec<usize> = (0..initial_indices.len()).collect();

    for &(left_orig, right_orig) in pairs {
        // Find current positions
        let left_pos = index_to_position[left_orig];
        let right_pos = index_to_position[right_orig];

        let (i, j) = if left_pos < right_pos {
            (left_pos, right_pos)
        } else {
            (right_pos, left_pos)
        };

        let indices_i: BTreeSet<char> = current_indices[i].iter().copied().collect();
        let indices_j: BTreeSet<char> = current_indices[j].iter().copied().collect();

        // Contracted indices
        let common: BTreeSet<char> = indices_i.intersection(&indices_j).copied().collect();
        let contracted: Vec<char> = common
            .iter()
            .filter(|c| !output_set.contains(c))
            .copied()
            .collect();
        let contracted_set: BTreeSet<char> = contracted.iter().copied().collect();

        // Result indices
        let mut result_indices: Vec<char> = Vec::new();
        let mut seen: BTreeSet<char> = BTreeSet::new();
        for &c in &current_indices[i] {
            if !contracted_set.contains(&c) && !seen.contains(&c) {
                result_indices.push(c);
                seen.insert(c);
            }
        }
        for &c in &current_indices[j] {
            if !contracted_set.contains(&c) && !seen.contains(&c) {
                result_indices.push(c);
                seen.insert(c);
            }
        }

        path.push(ContractionStep::new(
            (i, j),
            contracted,
            result_indices.clone(),
            0, // FLOPs computed elsewhere
        ));

        // Update state: remove j, replace i with result
        current_indices[i] = result_indices;
        current_indices.remove(j);

        // Update position mapping
        for pos in &mut index_to_position {
            if *pos == j {
                *pos = i;
            } else if *pos > j {
                *pos -= 1;
            }
        }
    }

    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_optimal_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];
        let cost_model = CostModel::default();

        let path = optimal_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_optimal_chain() {
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];
        let cost_model = CostModel::default();

        let path = optimal_path(&notation, shapes, &cost_model);

        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_optimal_skewed_chain() {
        // Skewed sizes should reveal optimal order matters
        // 2x10, 10x1000, 1000x3
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        let shapes: &[&[usize]] = &[&[2, 10], &[10, 1000], &[1000, 3]];
        let cost_model = CostModel { alpha: 0 }; // Pure FLOP cost

        let path = optimal_path(&notation, shapes, &cost_model);

        // Optimal should contract (0,1) first, then result with 2
        // or (1,2) first if that's cheaper
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_subsets_of_size() {
        let subs = subsets_of_size(4, 2);
        assert_eq!(subs.len(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_proper_subsets() {
        let set = 0b111u32; // {0, 1, 2}
        let subs: Vec<_> = proper_subsets(set).collect();
        // Should have 7 proper non-empty subsets
        assert_eq!(subs.len(), 7);
    }
}
