//! Diagonal extraction kernel.
//!
//! Extracts diagonal from a matrix: ii->i
//! Also supports batched: bii->bi

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Threads per block.
const BLOCK_SIZE: u32 = 256;

/// Launches the diagonal extraction kernel.
///
/// For input with shape [..., N, N], outputs [..., N].
pub fn launch_diagonal<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Must have at least 2 dimensions
    if input.shape.len() < 2 {
        return Err(EinsumError::launch("diagonal requires at least 2D input"));
    }

    let ndim = input.shape.len();
    let rows = input.shape[ndim - 2];
    let cols = input.shape[ndim - 1];

    // Must be square in the last two dimensions
    if rows != cols {
        return Err(EinsumError::launch(alloc::format!(
            "diagonal requires square matrix, got {}x{}",
            rows, cols
        )));
    }

    let n = rows; // Diagonal length
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
    let expected_output_size = batch_size * n;
    let output_size: usize = output.shape.iter().product();
    if output_size != expected_output_size {
        return Err(EinsumError::launch(alloc::format!(
            "diagonal output size mismatch: expected {}, got {}",
            expected_output_size, output_size
        )));
    }

    // Compute strides
    let row_stride = input.strides[ndim - 2];
    let col_stride = input.strides[ndim - 1];
    let diag_stride = row_stride + col_stride;
    let matrix_size = n * n;

    // Total diagonal elements
    let total_elements = batch_size * n;

    // Launch config
    let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };
    let num_cubes = (total_elements as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    unsafe {
        diagonal_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            input.as_arg(1),
            output.as_arg(1),
            ScalarArg::new(n as u32),
            ScalarArg::new(diag_stride as u32),
            ScalarArg::new(matrix_size as u32),
            ScalarArg::new(total_elements as u32),
            E::as_type_native_unchecked(),
        ).map_err(|e| EinsumError::launch(alloc::format!("diagonal kernel failed: {:?}", e)))
    }
}

#[cube(launch_unchecked)]
fn diagonal_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    n: u32,              // Diagonal length per matrix
    diag_stride: u32,    // Stride along diagonal
    matrix_size: u32,    // Elements per matrix
    total_elements: u32, // Total output elements
    #[define(E)] _dtype: StorageType,
) {
    let global_id = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;

    if global_id < total_elements {
        // Decompose into batch and diagonal index
        let batch_idx = global_id / n;
        let diag_idx = global_id % n;

        // Compute input index
        let input_idx = batch_idx * matrix_size + diag_idx * diag_stride;

        output[global_id] = input[input_idx];
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
