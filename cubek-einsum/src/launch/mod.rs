//! Launch module for einsum operations.
//!
//! Provides the high-level API for executing einsum operations on GPU.

mod config;
mod executor;
mod workspace;

pub use config::EinsumConfig;
pub use executor::einsum;
pub use workspace::Workspace;
