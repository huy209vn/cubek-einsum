//! # CubeK Einsum
//!
//! SOTA Einstein Summation (Einsum) implementation for GPU tensor operations.
//!
//! ## Features
//!
//! - Full einsum notation parsing with ellipsis support
//! - Optimal contraction path finding (greedy, DP, branch-and-bound)
//! - Pattern recognition for fast paths (matmul, reduce, transpose)
//! - Integration with optimized cubek kernels
//! - Autotuning support
//!
//! ## Example
//!
//! ```ignore
//! use cubek_einsum::{Einsum, einsum};
//!
//! // Matrix multiplication
//! let c = einsum!("ij,jk->ik", a, b);
//!
//! // Batched attention scores
//! let scores = einsum!("bhqd,bhkd->bhqk", queries, keys);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod error;
pub mod notation;
pub mod optimization;
pub mod pattern;
pub mod kernels;
pub mod launch;

pub use error::EinsumError;
pub use notation::{EinsumNotation, Subscript, parse_einsum};
pub use optimization::{ExecutionPlan, ExecutionStep, ContractionStrategy};
pub use pattern::{FastPath, PatternMatcher};
pub use launch::{einsum, EinsumConfig};
