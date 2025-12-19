//! Copy with reshape kernel for materializing permuted tensors.
//!
//! When a tensor is permuted via stride manipulation and then needs to be
//! reshaped (merging dimensions), we must copy the data to make it contiguous
//! in the permuted order before the reshape can work correctly.

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Block size for copy kernel.
const BLOCK_SIZE: u32 = 256;

/// Copies data from a potentially non-contiguous source to a contiguous destination.
///
/// The source tensor may have non-standard strides (from permutation), and this
/// function materializes it into a contiguous tensor with the destination's shape.
pub fn copy_reshape<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    // Validate total elements match
    let input_elements: usize = input.shape.iter().product();
    let output_elements: usize = output.shape.iter().product();

    if input_elements != output_elements {
        return Err(EinsumError::shape(alloc::format!(
            "copy_reshape: element count mismatch {} vs {}",
            input_elements, output_elements
        )));
    }

    if input_elements == 0 {
        return Ok(());
    }

    let num_cubes = ((input_elements as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cube_dim = CubeDim { x: BLOCK_SIZE, y: 1, z: 1 };
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    unsafe {
        copy_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            input.as_arg(1),
            output.as_arg(1),
            ScalarArg::new(input_elements as u32),
            E::as_type_native_unchecked(),
        ).map_err(|e| EinsumError::launch(alloc::format!("copy kernel failed: {:?}", e)))
    }
}

/// Copy kernel using ABSOLUTE_POS for coalesced access.
#[cube(launch_unchecked)]
fn copy_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    num_elements: u32,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS < num_elements {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}
