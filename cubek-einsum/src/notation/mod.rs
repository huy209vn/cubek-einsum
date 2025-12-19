//! Einsum notation parsing and representation.
//!
//! Supports the full einsum grammar:
//! - Basic: `ij,jk->ik`
//! - Ellipsis: `...ij,...jk->...ik`
//! - Implicit output: `ij,jk` (implies `->ik`)

mod parser;
mod subscript;
mod notation;
pub mod validation;

pub use parser::parse_einsum;
pub use subscript::{Subscript, Index};
pub use notation::EinsumNotation;
pub use validation::validate_notation;
