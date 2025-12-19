//! Trace kernel.
//!
//! Computes trace(A) = sum of diagonal elements.
//! For ii-> notation.

use alloc::vec;
use alloc::vec::Vec;

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use cubek_reduce::components::instructions::ReduceOperationConfig;
use cubek_reduce::launch::{LineSizeStrategy, RoutineStrategy};
use cubek_reduce::routines::{BlueprintStrategy, unit::UnitStrategy};
use cubek_reduce::ReduceStrategy;

use crate::error::{EinsumError, EinsumResult};
use super::diagonal::launch_diagonal;

/// Launches the trace kernel.
///
/// Computes `output = sum(input[i,i])` for a square matrix.
/// For higher-dimensional tensors, traces over the last two dimensions.
///
/// Implementation: extracts diagonal into workspace, then reduces.
pub fn launch_trace<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Must have at least 2 dimensions
    if input.shape.len() < 2 {
        return Err(EinsumError::launch("trace requires at least 2D input"));
    }

    let ndim = input.shape.len();
    let rows = input.shape[ndim - 2];
    let cols = input.shape[ndim - 1];

    // Must be square in the last two dimensions
    if rows != cols {
        return Err(EinsumError::launch(alloc::format!(
            "trace requires square matrix, got {}x{}",
            rows, cols
        )));
    }

    let n = rows;
    if n == 0 {
        return Ok(());
    }

    // Compute batch size
    let batch_size: usize = if ndim > 2 {
        input.shape[..ndim - 2].iter().product()
    } else {
        1
    };

    // Verify output shape
    let expected_output_size = batch_size;
    let output_size: usize = output.shape.iter().product();
    if output_size != expected_output_size {
        return Err(EinsumError::launch(alloc::format!(
            "trace output size mismatch: expected {}, got {}",
            expected_output_size, output_size
        )));
    }

    // Step 1: Allocate workspace for diagonal extraction
    // Diagonal shape: [...batch_dims..., n]
    let diagonal_shape: Vec<usize> = if ndim > 2 {
        let mut shape = input.shape[..ndim - 2].to_vec();
        shape.push(n);
        shape
    } else {
        vec![n]
    };

    let dtype = input.dtype;
    let mut diagonal_workspace = TensorHandle::zeros(client, diagonal_shape, dtype);

    // Step 2: Extract diagonal
    launch_diagonal::<R, E>(client, input, &mut diagonal_workspace)?;

    // Step 3: Reduce diagonal along last axis to get trace
    // For batched case: reduce axis = ndim - 2 (the diagonal length axis)
    // For non-batched: reduce axis = 0
    let reduce_axis = diagonal_workspace.shape.len() - 1;

    let operation = ReduceOperationConfig::Sum;
    let elem_type = dtype.elem_type();
    let dtypes = operation.precision(elem_type, None);

    cubek_reduce::reduce(
        client,
        diagonal_workspace.as_ref(),
        output.as_ref(),
        reduce_axis,
        ReduceStrategy {
            line_size: LineSizeStrategy { parallel_output_vectorization: false },
            routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        }, // Auto strategy
        operation,
        dtypes,
    ).map_err(|e| EinsumError::launch(alloc::format!("trace reduce failed: {:?}", e)))
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
