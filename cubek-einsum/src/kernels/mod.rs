//! GPU kernels for einsum operations.
//!
//! Contains implementations of:
//! - Element-wise operations (hadamard, outer product, broadcast multiply)
//! - Reduction operations (dot product, trace)
//! - Diagonal operations (extraction)
//! - Copy/reshape operations (for materializing permuted tensors)

mod hadamard;
mod outer_product;
mod broadcast_multiply;
mod dot_product;
mod trace;
mod diagonal;
mod copy_reshape;

pub use hadamard::launch_hadamard;
pub use outer_product::launch_outer_product;
pub use broadcast_multiply::launch_broadcast_multiply;
pub use dot_product::launch_dot_product;
pub use trace::launch_trace;
pub use diagonal::launch_diagonal;
pub use copy_reshape::copy_reshape;
