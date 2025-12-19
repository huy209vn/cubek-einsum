//! Broadcast multiply kernel.
//!
//! Handles element-wise multiplication with broadcasting support.
//! Unlike hadamard (which assumes contiguous memory), this kernel
//! properly handles strided tensors where broadcast dimensions have stride=0.

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Block size for broadcast multiply.
const BLOCK_SIZE: u32 = 256;

/// Launches the broadcast multiply kernel.
///
/// Computes `output = lhs * rhs` element-wise with proper stride handling.
/// Supports broadcasting where dimensions with stride=0 are broadcast.
///
/// Requirements:
/// - Output shape must be compatible with broadcasting rules
/// - All tensors must have the same rank (caller should pad shapes)
pub fn launch_broadcast_multiply<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    let rank = output.shape.len();

    if lhs.shape.len() != rank || rhs.shape.len() != rank {
        return Err(EinsumError::launch(alloc::format!(
            "broadcast multiply: all tensors must have same rank (output: {}, lhs: {}, rhs: {})",
            rank, lhs.shape.len(), rhs.shape.len()
        )));
    }

    let num_elements: usize = output.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    let num_cubes = ((num_elements as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    // Launch appropriate kernel based on rank
    // For rank 1-2: use direct broadcast multiply kernel
    // For rank 3+: materialize broadcasts to contiguous memory first (too many params for kernel)
    unsafe {
        match rank {
            1 => broadcast_multiply_1d::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                lhs.as_arg(1),
                rhs.as_arg(1),
                output.as_arg(1),
                ScalarArg::new(num_elements as u32),
                ScalarArg::new(output.shape[0] as u32),
                ScalarArg::new(lhs.strides[0] as u32),
                ScalarArg::new(rhs.strides[0] as u32),
                E::as_type_native_unchecked(),
            ),
            2 => broadcast_multiply_2d::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                lhs.as_arg(1),
                rhs.as_arg(1),
                output.as_arg(1),
                ScalarArg::new(num_elements as u32),
                ScalarArg::new(output.shape[0] as u32),
                ScalarArg::new(output.shape[1] as u32),
                ScalarArg::new(lhs.strides[0] as u32),
                ScalarArg::new(lhs.strides[1] as u32),
                ScalarArg::new(rhs.strides[0] as u32),
                ScalarArg::new(rhs.strides[1] as u32),
                E::as_type_native_unchecked(),
            ),
            // For rank 3+: materialize broadcasts then use hadamard
            _ => {
                return materialize_and_multiply::<R, E>(client, lhs, rhs, output);
            }
        }.map_err(|e| EinsumError::launch(alloc::format!("broadcast multiply kernel failed: {:?}", e)))
    }
}

/// Fallback for high-rank tensors: materialize broadcasts to contiguous memory, then multiply.
fn materialize_and_multiply<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Materialize lhs if it has broadcast dimensions (stride=0)
    let lhs_contiguous = if lhs.strides.iter().any(|&s| s == 0) {
        materialize_tensor::<R, E>(client, lhs, &output.shape)?
    } else {
        lhs.clone()
    };

    // Materialize rhs if it has broadcast dimensions (stride=0)
    let rhs_contiguous = if rhs.strides.iter().any(|&s| s == 0) {
        materialize_tensor::<R, E>(client, rhs, &output.shape)?
    } else {
        rhs.clone()
    };

    // Now both are contiguous with matching shapes, use hadamard
    super::launch_hadamard::<R, E>(client, &lhs_contiguous, &rhs_contiguous, output)
}

/// Materialize a tensor with broadcast dimensions into a contiguous buffer.
fn materialize_tensor<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    target_shape: &[usize],
) -> EinsumResult<TensorHandle<R>> {
    let num_elements: usize = target_shape.iter().product();
    let result = TensorHandle::zeros(client, target_shape.to_vec(), input.dtype);

    if num_elements == 0 {
        return Ok(result);
    }

    let num_cubes = ((num_elements as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    let rank = target_shape.len();

    // Use rank-specialized copy kernels
    unsafe {
        match rank {
            1 => copy_broadcast_1d::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                input.as_arg(1),
                result.as_arg(1),
                ScalarArg::new(num_elements as u32),
                ScalarArg::new(target_shape[0] as u32),
                ScalarArg::new(input.strides[0] as u32),
                E::as_type_native_unchecked(),
            ),
            2 => copy_broadcast_2d::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                input.as_arg(1),
                result.as_arg(1),
                ScalarArg::new(num_elements as u32),
                ScalarArg::new(target_shape[0] as u32),
                ScalarArg::new(target_shape[1] as u32),
                ScalarArg::new(input.strides[0] as u32),
                ScalarArg::new(input.strides[1] as u32),
                E::as_type_native_unchecked(),
            ),
            3 => copy_broadcast_3d::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                input.as_arg(1),
                result.as_arg(1),
                ScalarArg::new(num_elements as u32),
                ScalarArg::new(target_shape[0] as u32),
                ScalarArg::new(target_shape[1] as u32),
                ScalarArg::new(target_shape[2] as u32),
                ScalarArg::new(input.strides[0] as u32),
                ScalarArg::new(input.strides[1] as u32),
                ScalarArg::new(input.strides[2] as u32),
                E::as_type_native_unchecked(),
            ),
            _ => {
                return Err(EinsumError::launch(alloc::format!(
                    "broadcast multiply: rank {} exceeds max supported (3)",
                    rank
                )));
            }
        }.map_err(|e| EinsumError::launch(alloc::format!("copy broadcast kernel failed: {:?}", e)))?;
    }

    Ok(result)
}

/// 1D broadcast multiply kernel
#[cube(launch_unchecked)]
fn broadcast_multiply_1d<E: Numeric>(
    lhs: &Tensor<Line<E>>,
    rhs: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    _shape0: u32,
    lhs_stride0: u32,
    rhs_stride0: u32,
    #[define(E)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_elements {
        // For 1D, linear index == coordinate
        let coord0 = idx;

        // Compute memory offsets using strides
        let lhs_offset = coord0 * lhs_stride0;
        let rhs_offset = coord0 * rhs_stride0;

        output[idx] = lhs[lhs_offset] * rhs[rhs_offset];
    }
}

/// 2D broadcast multiply kernel
#[cube(launch_unchecked)]
fn broadcast_multiply_2d<E: Numeric>(
    lhs: &Tensor<Line<E>>,
    rhs: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    _shape0: u32,
    shape1: u32,
    lhs_stride0: u32,
    lhs_stride1: u32,
    rhs_stride0: u32,
    rhs_stride1: u32,
    #[define(E)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_elements {
        // Convert linear index to 2D coordinates (row-major)
        let coord0 = idx / shape1;
        let coord1 = idx % shape1;

        // Compute memory offsets using strides
        let lhs_offset = coord0 * lhs_stride0 + coord1 * lhs_stride1;
        let rhs_offset = coord0 * rhs_stride0 + coord1 * rhs_stride1;

        output[idx] = lhs[lhs_offset] * rhs[rhs_offset];
    }
}

// --- Copy broadcast kernels for materializing broadcast tensors ---

/// 1D copy broadcast kernel
#[cube(launch_unchecked)]
fn copy_broadcast_1d<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    _shape0: u32,
    input_stride0: u32,
    #[define(E)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_elements {
        let coord0 = idx;
        let input_offset = coord0 * input_stride0;
        output[idx] = input[input_offset];
    }
}

/// 2D copy broadcast kernel
#[cube(launch_unchecked)]
fn copy_broadcast_2d<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    _shape0: u32,
    shape1: u32,
    input_stride0: u32,
    input_stride1: u32,
    #[define(E)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_elements {
        let coord0 = idx / shape1;
        let coord1 = idx % shape1;
        let input_offset = coord0 * input_stride0 + coord1 * input_stride1;
        output[idx] = input[input_offset];
    }
}

/// 3D copy broadcast kernel
#[cube(launch_unchecked)]
fn copy_broadcast_3d<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    _shape0: u32,
    shape1: u32,
    shape2: u32,
    input_stride0: u32,
    input_stride1: u32,
    input_stride2: u32,
    #[define(E)] _dtype: StorageType,
) {
    let idx = ABSOLUTE_POS;
    if idx < num_elements {
        let stride12 = shape1 * shape2;
        let coord0 = idx / stride12;
        let rem = idx % stride12;
        let coord1 = rem / shape2;
        let coord2 = rem % shape2;
        let input_offset = coord0 * input_stride0 + coord1 * input_stride1 + coord2 * input_stride2;
        output[idx] = input[input_offset];
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
