//! Outer product kernel.
//!
//! Computes C[i,j] = A[i] * B[j] for vectors, generalizes to higher dimensions.

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Launch configuration for outer product.
const TILE_SIZE: u32 = 16;

/// Launches the outer product kernel.
///
/// Computes `output[i,j,...,k,l,...] = lhs[i,j,...] * rhs[k,l,...]`.
pub fn launch_outer_product<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    let lhs_size: usize = lhs.shape.iter().product();
    let rhs_size: usize = rhs.shape.iter().product();

    if lhs_size == 0 || rhs_size == 0 {
        return Ok(());
    }

    // Compute expected output size
    let expected_output_size = lhs_size * rhs_size;
    let output_size: usize = output.shape.iter().product();

    if output_size != expected_output_size {
        return Err(EinsumError::launch(alloc::format!(
            "outer product output size mismatch: expected {}, got {}",
            expected_output_size, output_size
        )));
    }

    // Launch config: tile over output dimensions
    let cube_dim = CubeDim { x: TILE_SIZE, y: TILE_SIZE, z: 1 };
    let cubes_x = (lhs_size as u32 + TILE_SIZE - 1) / TILE_SIZE;
    let cubes_y = (rhs_size as u32 + TILE_SIZE - 1) / TILE_SIZE;
    let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);

    // Launch kernel
    unsafe {
        outer_product_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            lhs.as_arg(1),
            rhs.as_arg(1),
            output.as_arg(1),
            ScalarArg::new(lhs_size as u32),
            ScalarArg::new(rhs_size as u32),
            E::as_type_native_unchecked(),
        ).map_err(|e| EinsumError::launch(alloc::format!("outer product kernel failed: {:?}", e)))
    }
}

#[cube(launch_unchecked)]
fn outer_product_kernel<E: Numeric>(
    lhs: &Tensor<Line<E>>,
    rhs: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    lhs_size: u32,
    rhs_size: u32,
    #[define(E)] _dtype: StorageType,
) {
    // 2D grid layout for outer product
    let i = CUBE_POS_X * TILE_SIZE + UNIT_POS_X;
    let j = CUBE_POS_Y * TILE_SIZE + UNIT_POS_Y;

    if i < lhs_size && j < rhs_size {
        // Output index: row-major order where lhs indices come first
        let output_idx = i * rhs_size + j;

        // Load values from input tensors
        let a = lhs[i];
        let b = rhs[j];

        // Compute outer product: C[i,j] = A[i] * B[j]
        output[output_idx] = a * b;
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
