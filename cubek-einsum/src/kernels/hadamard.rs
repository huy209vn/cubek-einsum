//! Hadamard (element-wise) product kernel.
//!
//! Computes C = A ⊙ B (element-wise multiplication).

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Block size for hadamard product.
const BLOCK_SIZE: u32 = 256;

/// Launches the hadamard (element-wise) product kernel.
///
/// Computes `output = lhs * rhs` element-wise.
/// Both inputs and output must have the same shape.
pub fn launch_hadamard<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Validate shapes match
    if lhs.shape != rhs.shape {
        return Err(EinsumError::launch("hadamard requires same shape inputs"));
    }
    if lhs.shape != output.shape {
        return Err(EinsumError::launch("hadamard output shape mismatch"));
    }

    let num_elements: usize = lhs.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    // Simple launch config - let CubeCL handle vectorization implicitly
    let num_cubes = ((num_elements as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    // Launch kernel
    unsafe {
        hadamard_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            lhs.as_arg(1),
            rhs.as_arg(1),
            output.as_arg(1),
            ScalarArg::new(num_elements as u32),
            E::as_type_native_unchecked(),
        ).map_err(|e| EinsumError::launch(alloc::format!("hadamard kernel failed: {:?}", e)))
    }
}

/// Hadamard kernel using CubeCL's ABSOLUTE_POS for optimal coalescing.
///
/// This pattern allows the compiler to prove memory access is coalesced across all backends:
/// - CUDA: Hardware warp coalescing
/// - WGPU/Vulkan: SPIR-V compiler can prove sequential access
/// - CubeCL handles vectorization implicitly via Line<E>
#[cube(launch_unchecked)]
fn hadamard_kernel<E: Numeric>(
    lhs: &Tensor<Line<E>>,
    rhs: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    #[define(E)] _dtype: StorageType,
) {
    // ABSOLUTE_POS = CUBE_POS * CUBE_DIM + UNIT_POS
    // Compiler can prove this gives coalesced access
    if ABSOLUTE_POS < num_elements {
        output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS];
    }
}

#[cfg(test)]
mod tests {
    // Integration tests require a runtime
}
