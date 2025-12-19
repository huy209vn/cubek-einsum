//! Dot product kernel.
//!
//! Computes scalar = sum(A ⊙ B) = A · B.
//! Implementation: Fused multiply-reduce in a single kernel pass using
//! block-level parallel reduction for high performance.

use alloc::vec;

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Block size for dot product reduction (must be power of 2).
const BLOCK_SIZE: u32 = 256;

/// Launches the dot product kernel.
///
/// Computes `output = sum(lhs * rhs)` as a scalar.
/// Both inputs must have the same shape.
///
/// Implementation: Fused multiply-reduce with block-level tree reduction.
/// Each block computes a partial sum, then a final reduction combines them.
pub fn launch_dot_product<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Validate shapes match
    if lhs.shape != rhs.shape {
        return Err(EinsumError::launch("dot product requires same shape inputs"));
    }

    let num_elements: usize = lhs.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    // Output should be scalar
    let output_size: usize = output.shape.iter().product();
    if output_size != 1 {
        return Err(EinsumError::launch(alloc::format!(
            "dot product output should be scalar, got size {}",
            output_size
        )));
    }

    let dtype = lhs.dtype;
    let num_elements_u32 = num_elements as u32;

    // Calculate number of blocks needed
    let num_blocks = (num_elements_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };

    if num_blocks == 1 {
        // Small input: single block writes directly to output
        let cube_count = CubeCount::Static(1, 1, 1);

        unsafe {
            dot_product_fused::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                lhs.as_arg(1),
                rhs.as_arg(1),
                output.as_arg(1),
                ScalarArg::new(num_elements_u32),
                BLOCK_SIZE,
                E::as_type_native_unchecked(),
            ).map_err(|e| EinsumError::launch(alloc::format!("dot product kernel failed: {:?}", e)))?;
        }
    } else {
        // Large input: multi-block reduction
        // Phase 1: Each block computes partial sum
        let partial_sums = TensorHandle::zeros(client, vec![num_blocks as usize], dtype);
        let cube_count = CubeCount::Static(num_blocks, 1, 1);

        unsafe {
            dot_product_partial::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                lhs.as_arg(1),
                rhs.as_arg(1),
                partial_sums.as_arg(1),
                ScalarArg::new(num_elements_u32),
                BLOCK_SIZE,
                E::as_type_native_unchecked(),
            ).map_err(|e| EinsumError::launch(alloc::format!("dot product partial failed: {:?}", e)))?;
        }

        // Phase 2: Reduce partial sums to final result
        let final_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if final_blocks == 1 {
            let cube_count = CubeCount::Static(1, 1, 1);

            unsafe {
                reduce_partial_sums::launch_unchecked::<R>(
                    client,
                    cube_count,
                    cube_dim,
                    partial_sums.as_arg(1),
                    output.as_arg(1),
                    ScalarArg::new(num_blocks),
                    BLOCK_SIZE,
                    E::as_type_native_unchecked(),
                ).map_err(|e| EinsumError::launch(alloc::format!("dot product final reduce failed: {:?}", e)))?;
            }
        } else {
            // Need another level of reduction (very large inputs > 16M elements)
            let level2_partials = TensorHandle::zeros(client, vec![final_blocks as usize], dtype);
            let cube_count = CubeCount::Static(final_blocks, 1, 1);

            unsafe {
                reduce_partial_sums_multi::launch_unchecked::<R>(
                    client,
                    cube_count,
                    cube_dim,
                    partial_sums.as_arg(1),
                    level2_partials.as_arg(1),
                    ScalarArg::new(num_blocks),
                    BLOCK_SIZE,
                    E::as_type_native_unchecked(),
                ).map_err(|e| EinsumError::launch(alloc::format!("dot product level2 failed: {:?}", e)))?;
            }

            // Final single-block reduction
            let cube_count = CubeCount::Static(1, 1, 1);
            unsafe {
                reduce_partial_sums::launch_unchecked::<R>(
                    client,
                    cube_count,
                    cube_dim,
                    level2_partials.as_arg(1),
                    output.as_arg(1),
                    ScalarArg::new(final_blocks),
                    BLOCK_SIZE,
                    E::as_type_native_unchecked(),
                ).map_err(|e| EinsumError::launch(alloc::format!("dot product final2 reduce failed: {:?}", e)))?;
            }
        }
    }

    Ok(())
}

/// Fused dot product kernel for small inputs (single block).
/// Each thread accumulates multiple elements, then tree reduction.
#[cube(launch_unchecked)]
fn dot_product_fused<N: Numeric>(
    lhs: &Tensor<Line<N>>,
    rhs: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    num_elements: u32,
    #[comptime] block_size: u32,
    #[define(N)] _dtype: StorageType,
) {
    // Shared memory for tree reduction (scalar elements, not lines)
    let mut shared = SharedMemory::<N>::new(block_size);
    shared[UNIT_POS] = N::from_int(0);

    // Each thread accumulates its portion with stride
    // Note: lhs[idx] returns Line<N>, we extract scalar via index [0]
    let mut idx = UNIT_POS;
    while idx < num_elements {
        let a = lhs[idx];
        let b = rhs[idx];
        let prod = a * b;
        shared[UNIT_POS] += prod[0];
        idx += CUBE_DIM;
    }

    // Tree reduction (block_size is power of 2)
    sync_cube();
    let mut num_active = block_size.runtime();
    let mut jump = 1u32;
    while num_active > 1 {
        num_active /= 2;
        let dest = jump * 2 * UNIT_POS;
        let src = jump * (2 * UNIT_POS + 1);
        if UNIT_POS < num_active {
            let val = shared[src];
            shared[dest] += val;
        }
        jump *= 2;
        sync_cube();
    }

    // Thread 0 writes the final result
    if UNIT_POS == 0 {
        output[0] = Line::new(shared[0]);
    }
}

/// Multi-block dot product - each block writes its partial sum.
#[cube(launch_unchecked)]
fn dot_product_partial<N: Numeric>(
    lhs: &Tensor<Line<N>>,
    rhs: &Tensor<Line<N>>,
    partial_sums: &mut Tensor<Line<N>>,
    num_elements: u32,
    #[comptime] block_size: u32,
    #[define(N)] _dtype: StorageType,
) {
    let mut shared = SharedMemory::<N>::new(block_size);
    shared[UNIT_POS] = N::from_int(0);

    // Grid-stride loop for coalesced access
    let grid_size = CUBE_COUNT * CUBE_DIM;
    let mut idx = CUBE_POS * CUBE_DIM + UNIT_POS;

    while idx < num_elements {
        let a = lhs[idx];
        let b = rhs[idx];
        let prod = a * b;
        shared[UNIT_POS] += prod[0];
        idx += grid_size;
    }

    // Tree reduction
    sync_cube();
    let mut num_active = block_size.runtime();
    let mut jump = 1u32;
    while num_active > 1 {
        num_active /= 2;
        let dest = jump * 2 * UNIT_POS;
        let src = jump * (2 * UNIT_POS + 1);
        if UNIT_POS < num_active {
            let val = shared[src];
            shared[dest] += val;
        }
        jump *= 2;
        sync_cube();
    }

    // Thread 0 writes block's partial sum
    if UNIT_POS == 0 {
        partial_sums[CUBE_POS] = Line::new(shared[0]);
    }
}

/// Single-block reduction of partial sums to final output.
#[cube(launch_unchecked)]
fn reduce_partial_sums<N: Numeric>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    num_elements: u32,
    #[comptime] block_size: u32,
    #[define(N)] _dtype: StorageType,
) {
    let mut shared = SharedMemory::<N>::new(block_size);
    shared[UNIT_POS] = N::from_int(0);

    // Load and accumulate
    let mut idx = UNIT_POS;
    while idx < num_elements {
        shared[UNIT_POS] += input[idx][0];
        idx += CUBE_DIM;
    }

    // Tree reduction
    sync_cube();
    let mut num_active = block_size.runtime();
    let mut jump = 1u32;
    while num_active > 1 {
        num_active /= 2;
        let dest = jump * 2 * UNIT_POS;
        let src = jump * (2 * UNIT_POS + 1);
        if UNIT_POS < num_active {
            let val = shared[src];
            shared[dest] += val;
        }
        jump *= 2;
        sync_cube();
    }

    if UNIT_POS == 0 {
        output[0] = Line::new(shared[0]);
    }
}

/// Multi-block reduction of partial sums.
#[cube(launch_unchecked)]
fn reduce_partial_sums_multi<N: Numeric>(
    input: &Tensor<Line<N>>,
    partial_sums: &mut Tensor<Line<N>>,
    num_elements: u32,
    #[comptime] block_size: u32,
    #[define(N)] _dtype: StorageType,
) {
    let mut shared = SharedMemory::<N>::new(block_size);
    shared[UNIT_POS] = N::from_int(0);

    let grid_size = CUBE_COUNT * CUBE_DIM;
    let mut idx = CUBE_POS * CUBE_DIM + UNIT_POS;

    while idx < num_elements {
        shared[UNIT_POS] += input[idx][0];
        idx += grid_size;
    }

    sync_cube();
    let mut num_active = block_size.runtime();
    let mut jump = 1u32;
    while num_active > 1 {
        num_active /= 2;
        let dest = jump * 2 * UNIT_POS;
        let src = jump * (2 * UNIT_POS + 1);
        if UNIT_POS < num_active {
            let val = shared[src];
            shared[dest] += val;
        }
        jump *= 2;
        sync_cube();
    }

    if UNIT_POS == 0 {
        partial_sums[CUBE_POS] = Line::new(shared[0]);
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
