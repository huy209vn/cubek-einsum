//! Execution plan for einsum operations.

use alloc::vec;
use alloc::vec::Vec;

use super::cost::CostModel;
use super::greedy::greedy_path;
use super::dynamic::{optimal_path, MAX_DP_TENSORS};
use super::branch_bound::branch_bound_path;
use super::path::ContractionPath;
use crate::notation::EinsumNotation;
use crate::pattern::FastPath;

/// Maximum tensors for branch and bound before fallback to greedy.
const MAX_BB_TENSORS: usize = 20;

/// Strategy for finding contraction paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractionStrategy {
    /// Greedy algorithm - fast O(n³) heuristic.
    Greedy,
    /// Optimal dynamic programming - exponential but optimal for small n.
    Optimal,
    /// Branch and bound - good balance for medium n.
    BranchBound,
    /// Automatically choose based on problem size.
    #[default]
    Auto,
}

/// A single step in the execution plan.
#[derive(Debug, Clone)]
pub enum ExecutionStep {
    /// Use a fast path (optimized primitive).
    FastPath(FastPath),
    /// Perform a general contraction.
    Contraction {
        /// Input tensor indices (in current tensor list).
        inputs: (usize, usize),
        /// Indices being contracted.
        contracted: Vec<char>,
        /// Result tensor indices.
        result: Vec<char>,
        /// Estimated FLOPs.
        flops: u64,
    },
    /// Perform a permutation (transpose).
    Permutation {
        /// Input tensor index.
        input: usize,
        /// Permutation of dimensions.
        perm: Vec<usize>,
    },
    /// Perform a reduction.
    Reduction {
        /// Input tensor index.
        input: usize,
        /// Axes to reduce.
        axes: Vec<usize>,
        /// Reduction operation.
        op: ReductionOp,
    },
}

/// Type of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Prod,
    Max,
    Min,
}

/// Complete execution plan for an einsum operation.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Steps to execute.
    steps: Vec<ExecutionStep>,
    /// Total estimated FLOPs.
    total_flops: u64,
    /// Output shape.
    output_shape: Vec<usize>,
    /// Whether a fast path is used.
    uses_fast_path: bool,
    /// Initial input indices from notation (for multi-step contractions).
    input_indices: Vec<Vec<char>>,
}

impl ExecutionPlan {
    /// Creates a plan that uses a fast path.
    pub fn fast_path(fast_path: FastPath, output_shape: Vec<usize>, flops: u64) -> Self {
        Self {
            steps: vec![ExecutionStep::FastPath(fast_path)],
            total_flops: flops,
            output_shape,
            uses_fast_path: true,
            input_indices: Vec::new(),
        }
    }

    /// Creates a plan from a contraction path.
    pub fn from_contraction_path(
        path: ContractionPath,
        output_shape: Vec<usize>,
        input_indices: Vec<Vec<char>>,
    ) -> Self {
        let total_flops = path.total_flops();
        let steps: Vec<ExecutionStep> = path
            .steps()
            .iter()
            .map(|step| ExecutionStep::Contraction {
                inputs: step.inputs,
                contracted: step.contracted_indices.clone(),
                result: step.result_indices.clone(),
                flops: step.estimated_flops,
            })
            .collect();

        Self {
            steps,
            total_flops,
            output_shape,
            uses_fast_path: false,
            input_indices,
        }
    }

    /// Returns the execution steps.
    pub fn steps(&self) -> &[ExecutionStep] {
        &self.steps
    }

    /// Returns the total estimated FLOPs.
    pub fn total_flops(&self) -> u64 {
        self.total_flops
    }

    /// Returns the output shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Returns whether a fast path is used.
    pub fn uses_fast_path(&self) -> bool {
        self.uses_fast_path
    }

    /// Returns the number of steps.
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Returns the initial input indices from the notation.
    pub fn input_indices(&self) -> &[Vec<char>] {
        &self.input_indices
    }
}

/// Creates an execution plan for an einsum operation.
///
/// This is the main entry point for planning. It:
/// 1. Checks for fast paths (matmul, reduce, etc.)
/// 2. If no fast path, finds optimal contraction order
/// 3. Returns a complete execution plan
pub fn create_plan(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
    strategy: ContractionStrategy,
) -> ExecutionPlan {
    // First, check for fast paths
    if let Some(fast_path) = crate::pattern::recognize_pattern(notation) {
        // Compute output shape
        let output_shape = compute_output_shape(notation, shapes);
        let flops = estimate_fast_path_flops(&fast_path, shapes);
        return ExecutionPlan::fast_path(fast_path, output_shape, flops);
    }

    // No fast path - use contraction path optimization
    let cost_model = CostModel::default();
    let n = notation.num_inputs();
    let path = match strategy {
        ContractionStrategy::Greedy => greedy_path(notation, shapes, &cost_model),
        ContractionStrategy::Optimal => {
            if n <= MAX_DP_TENSORS {
                optimal_path(notation, shapes, &cost_model)
            } else {
                greedy_path(notation, shapes, &cost_model)
            }
        }
        ContractionStrategy::BranchBound => {
            if n <= MAX_BB_TENSORS {
                branch_bound_path(notation, shapes, &cost_model)
            } else {
                greedy_path(notation, shapes, &cost_model)
            }
        }
        ContractionStrategy::Auto => {
            if n <= 4 {
                // Small problems: use DP for optimal solution
                optimal_path(notation, shapes, &cost_model)
            } else if n <= MAX_DP_TENSORS {
                // Medium problems: use branch and bound
                branch_bound_path(notation, shapes, &cost_model)
            } else if n <= MAX_BB_TENSORS {
                // Larger problems: still use branch and bound with pruning
                branch_bound_path(notation, shapes, &cost_model)
            } else {
                // Very large: fall back to greedy
                greedy_path(notation, shapes, &cost_model)
            }
        }
    };

    let output_shape = compute_output_shape(notation, shapes);

    // Extract input indices from notation for the executor
    let input_indices: Vec<Vec<char>> = notation
        .inputs()
        .iter()
        .map(|s| s.named_indices().collect())
        .collect();

    ExecutionPlan::from_contraction_path(path, output_shape, input_indices)
}

/// Computes the output shape from notation and input shapes.
fn compute_output_shape(notation: &EinsumNotation, shapes: &[&[usize]]) -> Vec<usize> {
    use hashbrown::HashMap;

    let mut dim_map: HashMap<char, usize> = HashMap::new();

    for (input, shape) in notation.inputs().iter().zip(shapes.iter()) {
        for (c, &d) in input.named_indices().zip(shape.iter()) {
            dim_map.insert(c, d);
        }
    }

    notation
        .output()
        .named_indices()
        .filter_map(|c| dim_map.get(&c).copied())
        .collect()
}

/// Estimates FLOPs for a fast path operation.
fn estimate_fast_path_flops(fast_path: &FastPath, shapes: &[&[usize]]) -> u64 {
    match fast_path {
        FastPath::Matmul { .. } => {
            if shapes.len() >= 2 {
                let m = shapes[0].get(0).copied().unwrap_or(1) as u64;
                let k = shapes[0].get(1).copied().unwrap_or(1) as u64;
                let n = shapes[1].get(1).copied().unwrap_or(1) as u64;
                2 * m * k * n
            } else {
                0
            }
        }
        FastPath::BatchedMatmul { batch_dims, .. } => {
            if shapes.len() >= 2 {
                let batch_size: u64 = batch_dims.iter().map(|&i| {
                    shapes[0].get(i).copied().unwrap_or(1) as u64
                }).product();
                let m = shapes[0].get(batch_dims.len()).copied().unwrap_or(1) as u64;
                let k = shapes[0].get(batch_dims.len() + 1).copied().unwrap_or(1) as u64;
                let n = shapes[1].get(batch_dims.len() + 1).copied().unwrap_or(1) as u64;
                2 * batch_size * m * k * n
            } else {
                0
            }
        }
        FastPath::Reduce { .. } => {
            if let Some(shape) = shapes.get(0) {
                shape.iter().map(|&d| d as u64).product()
            } else {
                0
            }
        }
        FastPath::Transpose { .. } => {
            // Transpose is memory-bound, not compute-bound
            if let Some(shape) = shapes.get(0) {
                shape.iter().map(|&d| d as u64).product()
            } else {
                0
            }
        }
        FastPath::Hadamard => {
            if let Some(shape) = shapes.get(0) {
                shape.iter().map(|&d| d as u64).product()
            } else {
                0
            }
        }
        FastPath::OuterProduct => {
            if shapes.len() >= 2 {
                let size_a: u64 = shapes[0].iter().map(|&d| d as u64).product();
                let size_b: u64 = shapes[1].iter().map(|&d| d as u64).product();
                size_a * size_b
            } else {
                0
            }
        }
        FastPath::DotProduct => {
            if let Some(shape) = shapes.get(0) {
                2 * shape.iter().map(|&d| d as u64).product::<u64>()
            } else {
                0
            }
        }
        FastPath::Trace => {
            if let Some(shape) = shapes.get(0) {
                shape.get(0).copied().unwrap_or(1) as u64
            } else {
                0
            }
        }
        FastPath::DiagonalExtract => {
            if let Some(shape) = shapes.get(0) {
                shape.get(0).copied().unwrap_or(1) as u64
            } else {
                0
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_create_plan_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let shapes: &[&[usize]] = &[&[100, 200], &[200, 300]];

        let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

        assert!(plan.uses_fast_path());
        assert_eq!(plan.output_shape(), &[100, 300]);
    }

    #[test]
    fn test_create_plan_chain() {
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        let shapes: &[&[usize]] = &[&[10, 20], &[20, 30], &[30, 40]];

        let plan = create_plan(&notation, shapes, ContractionStrategy::Auto);

        // Chain of 3 doesn't match fast path, should use contraction
        assert!(!plan.uses_fast_path());
        assert_eq!(plan.num_steps(), 2);
    }
}
