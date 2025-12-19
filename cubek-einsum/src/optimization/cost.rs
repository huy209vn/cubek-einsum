//! Cost model for contraction operations.

use alloc::vec::Vec;
use hashbrown::HashMap;

/// Cost of a single contraction operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContractionCost {
    /// Number of floating-point operations.
    pub flops: u64,
    /// Memory traffic in elements.
    pub memory: u64,
    /// Combined cost using the cost model.
    pub total: u64,
}

impl ContractionCost {
    pub fn new(flops: u64, memory: u64, alpha: u64) -> Self {
        let total = flops.saturating_add(memory.saturating_mul(alpha));
        Self { flops, memory, total }
    }

    pub fn zero() -> Self {
        Self { flops: 0, memory: 0, total: 0 }
    }
}

impl core::ops::Add for ContractionCost {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            flops: self.flops.saturating_add(rhs.flops),
            memory: self.memory.saturating_add(rhs.memory),
            total: self.total.saturating_add(rhs.total),
        }
    }
}

impl Ord for ContractionCost {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.total.cmp(&other.total)
    }
}

impl PartialOrd for ContractionCost {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ContractionCost {}

/// Cost model for evaluating contraction operations.
#[derive(Debug, Clone)]
pub struct CostModel {
    /// Memory bandwidth penalty factor.
    /// Higher values penalize memory-bound operations more.
    pub alpha: u64,
}

impl Default for CostModel {
    fn default() -> Self {
        // Default alpha suitable for GPUs where memory bandwidth is expensive
        Self { alpha: 64 }
    }
}

impl CostModel {
    /// Creates a cost model optimized for GPU execution.
    pub fn gpu() -> Self {
        Self { alpha: 64 }
    }

    /// Creates a cost model optimized for CPU execution.
    pub fn cpu() -> Self {
        Self { alpha: 8 }
    }

    /// Computes the cost of contracting two tensors.
    ///
    /// # Arguments
    /// * `shape_a` - Shape of first tensor
    /// * `shape_b` - Shape of second tensor
    /// * `indices_a` - Index characters for first tensor
    /// * `indices_b` - Index characters for second tensor
    /// * `contracted` - Indices being contracted (summed over)
    pub fn compute_pairwise_cost(
        &self,
        shape_a: &[usize],
        shape_b: &[usize],
        indices_a: &[char],
        indices_b: &[char],
        contracted: &[char],
    ) -> ContractionCost {
        // Build dimension map
        let mut dim_map: HashMap<char, usize> = HashMap::new();
        for (&c, &d) in indices_a.iter().zip(shape_a.iter()) {
            dim_map.insert(c, d);
        }
        for (&c, &d) in indices_b.iter().zip(shape_b.iter()) {
            dim_map.insert(c, d);
        }

        // Output indices: union minus contracted
        let contracted_set: hashbrown::HashSet<char> = contracted.iter().copied().collect();

        // Compute output size
        let mut output_size: u64 = 1;
        for &c in indices_a.iter().chain(indices_b.iter()) {
            if !contracted_set.contains(&c) {
                if let Some(&d) = dim_map.get(&c) {
                    // Only count each index once
                    output_size = output_size.saturating_mul(d as u64);
                }
            }
        }
        // Deduplicate: divide by indices that appear in both
        let indices_a_set: hashbrown::HashSet<char> = indices_a.iter().copied().collect();
        let indices_b_set: hashbrown::HashSet<char> = indices_b.iter().copied().collect();
        let common_non_contracted: Vec<char> = indices_a_set
            .intersection(&indices_b_set)
            .filter(|&c| !contracted_set.contains(c))
            .copied()
            .collect();
        for &c in &common_non_contracted {
            if let Some(&d) = dim_map.get(&c) {
                // These were counted twice, divide once
                output_size /= d as u64;
            }
        }

        // Compute contracted size
        let contracted_size: u64 = contracted
            .iter()
            .filter_map(|c| dim_map.get(c))
            .map(|&d| d as u64)
            .product();

        // FLOPs = output_size * contracted_size * 2
        let flops = output_size.saturating_mul(contracted_size).saturating_mul(2);

        // Memory = read inputs + write output
        let input_a_size: u64 = shape_a.iter().map(|&d| d as u64).product();
        let input_b_size: u64 = shape_b.iter().map(|&d| d as u64).product();
        let memory = input_a_size + input_b_size + output_size;

        ContractionCost::new(flops, memory, self.alpha)
    }

    /// Computes the cost of contracting multiple tensors into one.
    pub fn compute_multi_cost(
        &self,
        shapes: &[Vec<usize>],
        indices: &[Vec<char>],
        output_indices: &[char],
    ) -> ContractionCost {
        // Build dimension map
        let mut dim_map: HashMap<char, usize> = HashMap::new();
        for (shape, idx) in shapes.iter().zip(indices.iter()) {
            for (&c, &d) in idx.iter().zip(shape.iter()) {
                dim_map.insert(c, d);
            }
        }

        // Determine contracted indices
        let output_set: hashbrown::HashSet<char> = output_indices.iter().copied().collect();
        let all_indices: hashbrown::HashSet<char> = indices
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let contracted: Vec<char> = all_indices
            .difference(&output_set)
            .copied()
            .collect();

        // Compute sizes
        let output_size: u64 = output_indices
            .iter()
            .filter_map(|c| dim_map.get(c))
            .map(|&d| d as u64)
            .product();

        let contracted_size: u64 = contracted
            .iter()
            .filter_map(|c| dim_map.get(c))
            .map(|&d| d as u64)
            .product();

        let flops = output_size.saturating_mul(contracted_size).saturating_mul(2);

        let input_size: u64 = shapes
            .iter()
            .map(|s| s.iter().map(|&d| d as u64).product::<u64>())
            .sum();
        let memory = input_size + output_size;

        ContractionCost::new(flops, memory, self.alpha)
    }

    /// Estimates the remaining cost of contracting a set of tensors.
    /// Used as lower bound in branch-and-bound.
    pub fn optimistic_remaining_cost(
        &self,
        shapes: &[Vec<usize>],
        _indices: &[Vec<char>],
    ) -> ContractionCost {
        if shapes.len() <= 1 {
            return ContractionCost::zero();
        }

        // Lower bound: assume all remaining contractions can be done optimally
        // This is an underestimate, which is what we want for branch-and-bound
        let total_elements: u64 = shapes
            .iter()
            .map(|s| s.iter().map(|&d| d as u64).product::<u64>())
            .sum();

        // Minimum FLOPs: at least need to touch all elements
        let flops = total_elements;
        let memory = total_elements;

        ContractionCost::new(flops, memory, self.alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_cost() {
        let model = CostModel::default();

        // ij,jk->ik: A[100,200] @ B[200,300] -> C[100,300]
        let cost = model.compute_pairwise_cost(
            &[100, 200],
            &[200, 300],
            &['i', 'j'],
            &['j', 'k'],
            &['j'],
        );

        // FLOPs = 100 * 300 * 200 * 2 = 12,000,000
        assert_eq!(cost.flops, 12_000_000);
    }

    #[test]
    fn test_cost_ordering() {
        let cheap = ContractionCost::new(100, 10, 64);
        let expensive = ContractionCost::new(1000, 100, 64);

        assert!(cheap < expensive);
    }
}
