//! Contraction path optimization for einsum.
//!
//! Implements multiple strategies for finding optimal contraction orderings:
//! - Greedy: O(n³) fast heuristic
//! - Dynamic Programming: Optimal for small n
//! - Branch and Bound: Good balance for medium n

mod cost;
mod greedy;
mod dynamic;
mod branch_bound;
mod path;
mod plan;

pub use cost::{CostModel, ContractionCost};
pub use greedy::greedy_path;
pub use dynamic::optimal_path;
pub use branch_bound::branch_bound_path;
pub use path::{ContractionPath, ContractionStep};
pub use plan::{ExecutionPlan, ExecutionStep, ContractionStrategy, ReductionOp, create_plan};
