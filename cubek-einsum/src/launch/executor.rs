//! Einsum execution engine.
//!
//! Orchestrates parsing, optimization, and kernel dispatch.

use alloc::vec::Vec;
use alloc::vec;

use cubecl::prelude::*;
use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;

use cubek_matmul::launch::{Strategy as MatmulStrategy, MatmulInputHandle, launch};
use cubek_matmul::definition::{MatmulElems, MatmulElemType};
use cubek_reduce::components::instructions::ReduceOperationConfig;
use cubek_reduce::launch::{LineSizeStrategy, RoutineStrategy};
use cubek_reduce::routines::{BlueprintStrategy, unit::UnitStrategy};
use cubek_reduce::ReduceStrategy;

use crate::error::{EinsumError, EinsumResult};
use crate::notation::{parse_einsum, EinsumNotation, validate_notation};
use crate::notation::validation::validate_shapes;
use crate::optimization::{create_plan, ExecutionStep, ReductionOp};
use crate::pattern::FastPath;
use crate::kernels;
use super::config::EinsumConfig;

/// Executes an einsum operation.
///
/// # Arguments
/// * `client` - The compute client
/// * `notation` - Einsum notation string (e.g., "ij,jk->ik")
/// * `inputs` - Input tensor handles
/// * `output` - Output tensor handle
/// * `config` - Optional configuration
///
/// # Example
///
/// ```ignore
/// let output = einsum(
///     &client,
///     "ij,jk->ik",
///     &[&a, &b],
///     &mut c,
///     None,
/// )?;
/// ```
pub fn einsum<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    notation_str: &str,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    config: Option<EinsumConfig>,
) -> EinsumResult<()> {
    let config = config.unwrap_or_default();

    // Parse notation
    let notation = parse_einsum(notation_str)?;

    // Validate notation
    validate_notation(&notation)?;

    // Extract shapes
    let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape.as_slice()).collect();

    // Validate shapes if enabled
    if config.validate_shapes {
        let _ = validate_shapes(&notation, &shapes)?;
    }

    // Create execution plan
    let plan = create_plan(&notation, &shapes, config.strategy);

    // Execute plan
    execute_plan::<R, E>(client, &plan, inputs, output, &config)
}

/// Executes a pre-parsed einsum notation.
///
/// Useful when the same notation will be executed multiple times.
#[allow(dead_code)]
pub fn einsum_with_notation<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    notation: &EinsumNotation,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    config: Option<EinsumConfig>,
) -> EinsumResult<()> {
    let config = config.unwrap_or_default();

    // Extract shapes
    let shapes: Vec<&[usize]> = inputs.iter().map(|t| t.shape.as_slice()).collect();

    // Validate shapes if enabled
    if config.validate_shapes {
        let _ = validate_shapes(notation, &shapes)?;
    }

    // Create execution plan
    let plan = create_plan(notation, &shapes, config.strategy);

    // Execute plan
    execute_plan::<R, E>(client, &plan, inputs, output, &config)
}

/// Executes an execution plan.
fn execute_plan<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    plan: &crate::optimization::ExecutionPlan,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    config: &EinsumConfig,
) -> EinsumResult<()> {
    if plan.uses_fast_path() {
        // Single fast-path operation
        match &plan.steps()[0] {
            ExecutionStep::FastPath(fast_path) => {
                execute_fast_path::<R, E>(client, fast_path, inputs, output, config)
            }
            _ => Err(EinsumError::unsupported("invalid plan structure")),
        }
    } else {
        // General contraction path
        execute_contractions::<R, E>(client, plan, inputs, output, config)
    }
}

/// Executes a fast-path operation.
fn execute_fast_path<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    fast_path: &FastPath,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    _config: &EinsumConfig,
) -> EinsumResult<()> {
    match fast_path {
        FastPath::Matmul { transpose_a, transpose_b } => {
            execute_matmul::<R, E>(client, inputs, output, *transpose_a, *transpose_b, &[])
        }
        FastPath::BatchedMatmul { batch_dims, transpose_a, transpose_b } => {
            execute_matmul::<R, E>(client, inputs, output, *transpose_a, *transpose_b, batch_dims)
        }
        FastPath::Reduce { axes, .. } => {
            execute_reduce::<R, E>(client, inputs, output, axes)
        }
        FastPath::Transpose { permutation } => {
            execute_transpose::<R, E>(inputs, output, permutation)
        }
        FastPath::Hadamard => {
            execute_hadamard::<R, E>(client, inputs, output)
        }
        FastPath::OuterProduct => {
            execute_outer_product::<R, E>(client, inputs, output)
        }
        FastPath::DotProduct => {
            execute_dot_product::<R, E>(client, inputs, output)
        }
        FastPath::Trace => {
            execute_trace::<R, E>(client, inputs, output)
        }
        FastPath::DiagonalExtract => {
            execute_diagonal::<R, E>(client, inputs, output)
        }
    }
}

/// Executes matrix multiplication via cubek-matmul.
fn execute_matmul<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    transpose_a: bool,
    transpose_b: bool,
    _batch_dims: &[usize],
) -> EinsumResult<()> {
    if inputs.len() < 2 {
        return Err(EinsumError::unsupported("matmul requires 2 inputs"));
    }

    let mut lhs = inputs[0].clone();
    let mut rhs = inputs[1].clone();

    // Handle transposition by swapping the last two dimensions
    // cubek-matmul expects row-major layout, transposition is handled via strides
    if transpose_a {
        let ndim = lhs.shape.len();
        if ndim >= 2 {
            lhs.shape.swap(ndim - 2, ndim - 1);
            lhs.strides.swap(ndim - 2, ndim - 1);
        }
    }

    if transpose_b {
        let ndim = rhs.shape.len();
        if ndim >= 2 {
            rhs.shape.swap(ndim - 2, ndim - 1);
            rhs.strides.swap(ndim - 2, ndim - 1);
        }
    }

    // Create element type
    let elem_type = MatmulElemType::new(E::as_type_native_unchecked(), false);

    // Create MatmulElems from single dtype (all same type)
    let dtypes = MatmulElems::from_single_dtype(elem_type);

    // Create input handles
    let lhs_handle = MatmulInputHandle::Normal(lhs);
    let rhs_handle = MatmulInputHandle::Normal(rhs);

    // Use Auto strategy for best performance
    let strategy = MatmulStrategy::Auto;

    // Launch matmul
    launch(
        &strategy,
        client,
        lhs_handle,
        rhs_handle,
        output.clone(),
        dtypes,
    ).map_err(|e| EinsumError::launch(alloc::format!("matmul failed: {:?}", e)))
}

/// Executes reduction via cubek-reduce.
///
/// Note: cubek_reduce expects output to keep reduced dimension with size 1,
/// but einsum semantics remove the dimension entirely. We handle this by
/// creating an intermediate tensor if needed.
fn execute_reduce<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    axes: &[usize],
) -> EinsumResult<()> {
    if inputs.is_empty() {
        return Err(EinsumError::unsupported("reduce requires at least 1 input"));
    }

    let input = inputs[0];

    // For multiple axes, we need to reduce sequentially
    if axes.len() > 1 {
        return execute_multi_axis_reduce::<R, E>(client, input, output, axes);
    }

    if axes.is_empty() {
        // No reduction needed, just copy
        return Err(EinsumError::unsupported("empty reduction axes"));
    }

    let axis = axes[0];

    // Get optimal precision for sum operation
    let operation = ReduceOperationConfig::Sum;
    let elem_type = E::as_type_native_unchecked().elem_type();
    let dtypes = operation.precision(elem_type, None);

    // cubek_reduce expects output shape to keep reduced dim with size 1
    // e.g., input [1024, 1024] reduced on axis 1 -> output [1024, 1]
    // But einsum expects [1024] (dim removed). Create intermediate if needed.
    let mut keep_dim_shape = input.shape.clone();
    keep_dim_shape[axis] = 1;

    let output_matches_keepdim = output.shape == keep_dim_shape;

    if output_matches_keepdim {
        // Output already has keep-dim shape, reduce directly
        cubek_reduce::reduce(
            client,
            input.as_ref(),
            output.as_ref(),
            axis,
            ReduceStrategy {
                line_size: LineSizeStrategy { parallel_output_vectorization: false },
                routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
            },
            operation,
            dtypes,
        ).map_err(|e| EinsumError::launch(alloc::format!("reduce failed: {:?}", e)))
    } else {
        // Create intermediate with keep-dim shape, then copy/reshape to output
        let intermediate = TensorHandle::zeros(client, keep_dim_shape.clone(), input.dtype);

        cubek_reduce::reduce(
            client,
            input.as_ref(),
            intermediate.as_ref(),
            axis,
            ReduceStrategy {
                line_size: LineSizeStrategy { parallel_output_vectorization: false },
                routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
            },
            operation,
            dtypes,
        ).map_err(|e| EinsumError::launch(alloc::format!("reduce failed: {:?}", e)))?;

        // The intermediate and output have the same number of elements,
        // just different shapes. We can treat them as the same memory.
        // Update output to point to intermediate's data with squeezed shape.
        output.handle = intermediate.handle;
        output.strides = compute_strides(&output.shape);

        Ok(())
    }
}

/// Executes multi-axis reduction by reducing one axis at a time.
///
/// Note: cubek_reduce keeps reduced dimensions as size 1, so we need to
/// handle shape differences between intermediate and final output.
fn execute_multi_axis_reduce<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    input: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
    axes: &[usize],
) -> EinsumResult<()> {
    // Sort axes in descending order so we can reduce without invalidating indices
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort_by(|a, b| b.cmp(a));

    let operation = ReduceOperationConfig::Sum;
    let elem_type = E::as_type_native_unchecked().elem_type();
    let dtype = input.dtype;
    let dtypes = operation.precision(elem_type, None);

    let mut current = input.clone();

    for (i, &axis) in sorted_axes.iter().enumerate() {
        let is_last = i == sorted_axes.len() - 1;

        // cubek_reduce expects keep-dim shape (dim becomes 1, not removed)
        let mut keep_dim_shape = current.shape.clone();
        keep_dim_shape[axis] = 1;

        if is_last {
            // Final reduction - check if output shape matches keep-dim
            if output.shape == keep_dim_shape {
                // Direct reduction to output
                cubek_reduce::reduce(
                    client,
                    current.as_ref(),
                    output.as_ref(),
                    axis,
                    ReduceStrategy {
                line_size: LineSizeStrategy { parallel_output_vectorization: false },
                routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
            },
                    operation,
                    dtypes,
                ).map_err(|e| EinsumError::launch(alloc::format!("reduce failed: {:?}", e)))?;
            } else {
                // Need intermediate, then copy to squeezed output
                let intermediate = TensorHandle::zeros(client, keep_dim_shape.clone(), dtype);

                cubek_reduce::reduce(
                    client,
                    current.as_ref(),
                    intermediate.as_ref(),
                    axis,
                    ReduceStrategy {
                line_size: LineSizeStrategy { parallel_output_vectorization: false },
                routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
            },
                    operation,
                    dtypes,
                ).map_err(|e| EinsumError::launch(alloc::format!("reduce failed: {:?}", e)))?;

                // Copy handle to output (same data, different shape interpretation)
                output.handle = intermediate.handle;
                output.strides = compute_strides(&output.shape);
            }
        } else {
            // Intermediate reduction - use keep-dim shape
            let workspace = TensorHandle::zeros(client, keep_dim_shape.clone(), dtype);

            cubek_reduce::reduce(
                client,
                current.as_ref(),
                workspace.as_ref(),
                axis,
                ReduceStrategy {
                line_size: LineSizeStrategy { parallel_output_vectorization: false },
                routine: RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
            },
                operation,
                dtypes,
            ).map_err(|e| EinsumError::launch(alloc::format!("reduce failed: {:?}", e)))?;

            // Use workspace as input for next iteration
            current = workspace;
        }
    }

    Ok(())
}

/// Computes strides for a given shape (row-major order).
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Executes transpose operation (zero-copy via stride manipulation).
fn execute_transpose<R: Runtime, E: CubePrimitive + Numeric>(
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    permutation: &[usize],
) -> EinsumResult<()> {
    if inputs.is_empty() {
        return Err(EinsumError::unsupported("transpose requires 1 input"));
    }

    let input = inputs[0];

    // Apply permutation to shape and strides
    let new_shape: Vec<usize> = permutation.iter().map(|&i| input.shape[i]).collect();
    let new_strides: Vec<usize> = permutation.iter().map(|&i| input.strides[i]).collect();

    // Update output metadata
    // Note: This assumes output shares the same underlying buffer as input
    // For a true einsum operation, the caller should set up output appropriately
    output.shape = new_shape;
    output.strides = new_strides;

    Ok(())
}

/// Executes Hadamard (element-wise) product.
fn execute_hadamard<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    if inputs.len() < 2 {
        return Err(EinsumError::unsupported("hadamard requires 2 inputs"));
    }
    kernels::launch_hadamard::<R, E>(client, inputs[0], inputs[1], output)
}

/// Executes broadcast multiply when no indices are contracted.
///
/// Handles patterns like `ij,j->ij` where one tensor broadcasts over the other.
/// This is NOT a matmul - it's element-wise multiplication with broadcasting.
fn execute_broadcast_multiply<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
    lhs_indices: &[char],
    rhs_indices: &[char],
) -> EinsumResult<()> {
    // For broadcast multiply, we need to expand tensors to match the output shape
    // and then do element-wise multiplication with proper stride handling.
    //
    // Example: ij,j->ij with shapes [512, 1024], [1024] -> [512, 1024]
    // The vector needs to be broadcast across the first dimension (stride=0 for that dim).

    let lhs_rank = lhs_indices.len();
    let rhs_rank = rhs_indices.len();

    // If shapes already match (same indices in same order), use Hadamard directly
    // (contiguous case, no broadcasting needed)
    if lhs_indices == rhs_indices && lhs.shape == rhs.shape {
        return kernels::launch_hadamard::<R, E>(client, lhs, rhs, output);
    }

    // For broadcast cases, we expand both tensors to match output shape/indices.
    // The output indices should be the union, preserving order from the larger tensor.

    // Determine output indices (union of both, preserving order)
    let output_indices: Vec<char> = if lhs_rank >= rhs_rank {
        lhs_indices.to_vec()
    } else {
        rhs_indices.to_vec()
    };
    let output_rank = output_indices.len();

    // Simple case: one tensor's indices are a subset of the other's
    // Example: ij,j->ij means rhs broadcasts over i
    if rhs_rank < lhs_rank && rhs_indices.iter().all(|c| lhs_indices.contains(c)) {
        // rhs broadcasts to match lhs (output)
        let mut rhs_broadcast = rhs.clone();

        // Expand rhs to match output shape with stride 0 for broadcast dims
        let mut new_shape = Vec::with_capacity(output_rank);
        let mut new_strides = Vec::with_capacity(output_rank);

        for (i, &idx) in lhs_indices.iter().enumerate() {
            if let Some(pos) = rhs_indices.iter().position(|&c| c == idx) {
                // This dimension exists in rhs
                new_shape.push(rhs.shape[pos]);
                new_strides.push(rhs.strides[pos]);
            } else {
                // Broadcast dimension - size from lhs/output, stride 0
                new_shape.push(lhs.shape[i]);
                new_strides.push(0);
            }
        }

        rhs_broadcast.shape = new_shape;
        rhs_broadcast.strides = new_strides;

        // Use the broadcast-aware kernel (handles strided tensors)
        return kernels::launch_broadcast_multiply::<R, E>(client, lhs, &rhs_broadcast, output);
    }

    // Symmetric case: lhs broadcasts to match rhs (output)
    if lhs_rank < rhs_rank && lhs_indices.iter().all(|c| rhs_indices.contains(c)) {
        let mut lhs_broadcast = lhs.clone();

        let mut new_shape = Vec::with_capacity(output_rank);
        let mut new_strides = Vec::with_capacity(output_rank);

        for (i, &idx) in rhs_indices.iter().enumerate() {
            if let Some(pos) = lhs_indices.iter().position(|&c| c == idx) {
                new_shape.push(lhs.shape[pos]);
                new_strides.push(lhs.strides[pos]);
            } else {
                new_shape.push(rhs.shape[i]);
                new_strides.push(0);
            }
        }

        lhs_broadcast.shape = new_shape;
        lhs_broadcast.strides = new_strides;

        // Use the broadcast-aware kernel (handles strided tensors)
        return kernels::launch_broadcast_multiply::<R, E>(client, &lhs_broadcast, rhs, output);
    }

    // Complex broadcast case - not yet supported
    Err(EinsumError::unsupported(alloc::format!(
        "complex broadcast multiply not yet supported: {:?} x {:?}",
        lhs_indices, rhs_indices
    )))
}

/// Executes outer product.
fn execute_outer_product<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    if inputs.len() < 2 {
        return Err(EinsumError::unsupported("outer product requires 2 inputs"));
    }
    kernels::launch_outer_product::<R, E>(client, inputs[0], inputs[1], output)
}

/// Executes dot product (reduction after element-wise multiply).
fn execute_dot_product<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    if inputs.len() < 2 {
        return Err(EinsumError::unsupported("dot product requires 2 inputs"));
    }
    kernels::launch_dot_product::<R, E>(client, inputs[0], inputs[1], output)
}

/// Executes trace (sum of diagonal).
fn execute_trace<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    if inputs.is_empty() {
        return Err(EinsumError::unsupported("trace requires 1 input"));
    }
    kernels::launch_trace::<R, E>(client, inputs[0], output)
}

/// Executes diagonal extraction.
fn execute_diagonal<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
) -> EinsumResult<()> {
    if inputs.is_empty() {
        return Err(EinsumError::unsupported("diagonal extraction requires 1 input"));
    }
    kernels::launch_diagonal::<R, E>(client, inputs[0], output)
}

/// Tracks a tensor with its index labels for contraction.
struct TrackedTensor<R: Runtime> {
    tensor: TensorHandle<R>,
    indices: Vec<char>,
}

/// Executes a general contraction sequence.
fn execute_contractions<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    plan: &crate::optimization::ExecutionPlan,
    inputs: &[&TensorHandle<R>],
    output: &mut TensorHandle<R>,
    config: &EinsumConfig,
) -> EinsumResult<()> {
    let steps = plan.steps();
    if steps.is_empty() {
        return Ok(());
    }

    // Get actual indices from the plan (parsed from notation)
    let plan_indices = plan.input_indices();

    // Initialize tracked tensors with input tensors and their actual indices from notation
    let mut tracked: Vec<TrackedTensor<R>> = inputs.iter().enumerate().map(|(idx, t)| {
        // Use indices from plan if available, otherwise infer
        let indices = if idx < plan_indices.len() {
            plan_indices[idx].clone()
        } else {
            infer_input_indices(idx, t.shape.len())
        };
        TrackedTensor {
            tensor: (*t).clone(),
            indices,
        }
    }).collect();
    let dtype = inputs.get(0).map(|t| t.dtype).unwrap_or_else(|| E::as_type_native_unchecked());

    for (step_idx, step) in steps.iter().enumerate() {
        let is_last = step_idx == steps.len() - 1;

        match step {
            ExecutionStep::Contraction { inputs: (i, j), contracted, result, .. } => {
                let i = *i;
                let j = *j;

                if i >= tracked.len() || j >= tracked.len() {
                    return Err(EinsumError::launch(alloc::format!(
                        "contraction step {} references invalid tensor indices ({}, {}), list has {} tensors",
                        step_idx, i, j, tracked.len()
                    )));
                }

                // Get indices for both tensors
                let lhs_indices = tracked[i].indices.clone();
                let rhs_indices = tracked[j].indices.clone();

                // Compute output shape using actual indices
                let contraction_output_shape = compute_contraction_shape_with_indices(
                    &tracked[i].tensor,
                    &tracked[j].tensor,
                    &lhs_indices,
                    &rhs_indices,
                    result,
                );

                // Determine where to write output
                if is_last {
                    execute_general_contraction_with_indices::<R, E>(
                        client,
                        &tracked[i].tensor,
                        &tracked[j].tensor,
                        output,
                        &lhs_indices,
                        &rhs_indices,
                        contracted,
                    )?;
                } else {
                    let mut workspace = TensorHandle::zeros(client, contraction_output_shape, dtype);

                    execute_general_contraction_with_indices::<R, E>(
                        client,
                        &tracked[i].tensor,
                        &tracked[j].tensor,
                        &mut workspace,
                        &lhs_indices,
                        &rhs_indices,
                        contracted,
                    )?;

                    // Update tracked list: remove i and j (higher index first), add result
                    let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
                    tracked.remove(max_idx);
                    tracked.remove(min_idx);
                    tracked.push(TrackedTensor {
                        tensor: workspace,
                        indices: result.to_vec(),
                    });
                }
            }
            ExecutionStep::FastPath(fast_path) => {
                return execute_fast_path::<R, E>(
                    client,
                    fast_path,
                    inputs,
                    output,
                    config,
                );
            }
            ExecutionStep::Permutation { input, perm } => {
                if *input >= tracked.len() {
                    return Err(EinsumError::launch("permutation references invalid tensor"));
                }
                let tracked_tensor = &mut tracked[*input];
                let new_shape: Vec<usize> = perm.iter().map(|&p| tracked_tensor.tensor.shape[p]).collect();
                let new_strides: Vec<usize> = perm.iter().map(|&p| tracked_tensor.tensor.strides[p]).collect();
                let new_indices: Vec<char> = perm.iter().map(|&p| tracked_tensor.indices[p]).collect();
                tracked_tensor.tensor.shape = new_shape;
                tracked_tensor.tensor.strides = new_strides;
                tracked_tensor.indices = new_indices;
            }
            ExecutionStep::Reduction { input, axes, op } => {
                if *input >= tracked.len() {
                    return Err(EinsumError::launch("reduction references invalid tensor"));
                }

                if *op != ReductionOp::Sum {
                    return Err(EinsumError::unsupported("only sum reduction supported"));
                }

                let tracked_tensor = &tracked[*input];

                if is_last {
                    execute_reduce::<R, E>(client, &[&tracked_tensor.tensor], output, axes)?;
                } else {
                    let mut reduced_shape: Vec<usize> = tracked_tensor.tensor.shape.iter()
                        .enumerate()
                        .filter(|(idx, _)| !axes.contains(idx))
                        .map(|(_, &d)| d)
                        .collect();
                    let reduced_indices: Vec<char> = tracked_tensor.indices.iter()
                        .enumerate()
                        .filter(|(idx, _)| !axes.contains(idx))
                        .map(|(_, &c)| c)
                        .collect();
                    if reduced_shape.is_empty() {
                        reduced_shape.push(1);
                    }

                    let mut workspace = TensorHandle::zeros(client, reduced_shape, dtype);
                    execute_reduce::<R, E>(client, &[&tracked_tensor.tensor], &mut workspace, axes)?;

                    tracked[*input] = TrackedTensor {
                        tensor: workspace,
                        indices: reduced_indices,
                    };
                }
            }
        }
    }

    Ok(())
}

/// Infers indices for input tensor based on position.
fn infer_input_indices(input_idx: usize, ndim: usize) -> Vec<char> {
    // Standard einsum index convention: a-z for first tensor, continuing for subsequent
    let all_indices: Vec<char> = ('a'..='z').collect();
    let start = input_idx * 4; // Assume up to 4 dims per tensor
    (0..ndim)
        .map(|i| all_indices.get(start + i).copied().unwrap_or('?'))
        .collect()
}

/// Computes the output shape for a contraction step with explicit indices.
fn compute_contraction_shape_with_indices<R: Runtime>(
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    lhs_indices: &[char],
    rhs_indices: &[char],
    result_indices: &[char],
) -> Vec<usize> {
    use hashbrown::HashMap;

    // Build dimension map from both inputs using their actual indices
    let mut dim_map: HashMap<char, usize> = HashMap::new();

    for (&idx, &size) in lhs_indices.iter().zip(lhs.shape.iter()) {
        dim_map.insert(idx, size);
    }
    for (&idx, &size) in rhs_indices.iter().zip(rhs.shape.iter()) {
        dim_map.insert(idx, size);
    }

    // Build output shape from result indices
    result_indices
        .iter()
        .filter_map(|c| dim_map.get(c).copied())
        .collect()
}

/// Executes a general two-tensor contraction with explicit index tracking.
///
/// Uses an optimized batched GEMM approach:
/// 1. Identify batch dimensions (indices in both LHS and RHS, not contracted)
/// 2. Permute tensors to [batch..., non-contracted..., contracted...] layout
/// 3. Reshape preserving batch structure: [batch..., M, K] and [batch..., K, N]
/// 4. Perform batched GEMM (cubek-matmul handles batches efficiently)
/// 5. Output already has correct shape from permutation
///
/// This is superior to flattening batch dims into M, as it:
/// - Preserves cache locality (batch as outer loop)
/// - Enables efficient batched kernel dispatch
/// - Handles strided tensors correctly
fn execute_general_contraction_with_indices<R: Runtime, E: CubePrimitive + Numeric>(
    client: &ComputeClient<R>,
    lhs: &TensorHandle<R>,
    rhs: &TensorHandle<R>,
    output: &mut TensorHandle<R>,
    lhs_indices: &[char],
    rhs_indices: &[char],
    contracted: &[char],
) -> EinsumResult<()> {
    use hashbrown::{HashMap, HashSet};

    // If no indices are contracted, this is a broadcast multiply, not a matmul
    // Fall back to element-wise kernel with broadcasting
    if contracted.is_empty() {
        return execute_broadcast_multiply::<R, E>(client, lhs, rhs, output, lhs_indices, rhs_indices);
    }

    // Build dimension map from index char to size
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (&idx, &size) in lhs_indices.iter().zip(lhs.shape.iter()) {
        dim_map.insert(idx, size);
    }
    for (&idx, &size) in rhs_indices.iter().zip(rhs.shape.iter()) {
        dim_map.insert(idx, size);
    }

    let contracted_set: HashSet<char> = contracted.iter().copied().collect();

    // Identify batch dimensions: indices that appear in both LHS and RHS but are not contracted
    let lhs_set: HashSet<char> = lhs_indices.iter().copied().collect();
    let rhs_set: HashSet<char> = rhs_indices.iter().copied().collect();
    let common_indices: HashSet<char> = lhs_set.intersection(&rhs_set).copied().collect();
    let batch_indices: Vec<char> = common_indices
        .difference(&contracted_set)
        .copied()
        .collect();
    let batch_set: HashSet<char> = batch_indices.iter().copied().collect();

    // For LHS: separate batch, non-contracted (M), and contracted (K) dimensions
    // We want order: [batch..., M..., K...]
    let lhs_batch: Vec<usize> = lhs_indices.iter()
        .enumerate()
        .filter(|(_, c)| batch_set.contains(*c))
        .map(|(i, _)| i)
        .collect();
    let lhs_m: Vec<usize> = lhs_indices.iter()
        .enumerate()
        .filter(|(_, c)| !contracted_set.contains(*c) && !batch_set.contains(*c))
        .map(|(i, _)| i)
        .collect();
    let lhs_contracted: Vec<usize> = lhs_indices.iter()
        .enumerate()
        .filter(|(_, c)| contracted_set.contains(*c))
        .map(|(i, _)| i)
        .collect();

    // For RHS: separate batch, contracted (K), and non-contracted (N) dimensions
    // We want order: [batch..., K..., N...]
    let rhs_batch: Vec<usize> = rhs_indices.iter()
        .enumerate()
        .filter(|(_, c)| batch_set.contains(*c))
        .map(|(i, _)| i)
        .collect();

    // CRITICAL: RHS contracted dimensions must be in the same ORDER as LHS contracted
    // dimensions for the merged K dimension to match. We build a map from contracted
    // index char to position in RHS, then order by LHS contracted order.
    let rhs_contracted_map: HashMap<char, usize> = rhs_indices.iter()
        .enumerate()
        .filter(|(_, c)| contracted_set.contains(*c))
        .map(|(i, &c)| (c, i))
        .collect();

    // Get contracted chars in the order they appear in LHS
    let lhs_contracted_chars: Vec<char> = lhs_indices.iter()
        .filter(|c| contracted_set.contains(*c))
        .copied()
        .collect();

    // Order RHS contracted positions to match LHS contracted order
    let rhs_contracted: Vec<usize> = lhs_contracted_chars.iter()
        .filter_map(|c| rhs_contracted_map.get(c).copied())
        .collect();

    let rhs_n: Vec<usize> = rhs_indices.iter()
        .enumerate()
        .filter(|(_, c)| !contracted_set.contains(*c) && !batch_set.contains(*c))
        .map(|(i, _)| i)
        .collect();

    // Compute batch dimensions, M, K, N
    let batch_shape: Vec<usize> = batch_indices.iter()
        .map(|c| dim_map[c])
        .collect();

    let m: usize = lhs_m.iter()
        .map(|&i| lhs.shape[i])
        .product::<usize>()
        .max(1);
    let k: usize = lhs_contracted.iter()
        .map(|&i| lhs.shape[i])
        .product::<usize>()
        .max(1);
    let n: usize = rhs_n.iter()
        .map(|&i| rhs.shape[i])
        .product::<usize>()
        .max(1);

    // Build permutation for LHS: [batch..., M..., K...]
    let lhs_perm: Vec<usize> = lhs_batch.iter()
        .chain(lhs_m.iter())
        .chain(lhs_contracted.iter())
        .copied()
        .collect();

    // Build permutation for RHS: [batch..., K..., N...]
    let rhs_perm: Vec<usize> = rhs_batch.iter()
        .chain(rhs_contracted.iter())
        .chain(rhs_n.iter())
        .copied()
        .collect();

    // Check if LHS needs transposition (permutation is not identity)
    let lhs_needs_transpose = !is_identity_permutation(&lhs_perm);
    let rhs_needs_transpose = !is_identity_permutation(&rhs_perm);

    // Determine target shapes for matmul
    let lhs_target_shape = if !batch_shape.is_empty() {
        [batch_shape.clone(), vec![m, k]].concat()
    } else {
        vec![m, k]
    };
    let rhs_target_shape = if !batch_shape.is_empty() {
        [batch_shape.clone(), vec![k, n]].concat()
    } else {
        vec![k, n]
    };

    // Check if reshape is needed (merging multiple dimensions)
    let lhs_needs_reshape = lhs_perm.len() != lhs_target_shape.len();
    let rhs_needs_reshape = rhs_perm.len() != rhs_target_shape.len();

    // For permute+reshape, we need to materialize the permuted tensor
    // because reshape requires contiguous data in the permuted order
    let lhs_reshaped = if lhs_needs_transpose && lhs_needs_reshape {
        // Need to materialize: copy data in permuted order, then reshape
        let permuted_shape: Vec<usize> = lhs_perm.iter().map(|&i| lhs.shape[i]).collect();
        let permuted_strides: Vec<usize> = lhs_perm.iter().map(|&i| lhs.strides[i]).collect();

        // Create permuted view
        let mut permuted = lhs.clone();
        permuted.shape = permuted_shape;
        permuted.strides = permuted_strides;

        // Materialize into contiguous tensor with target shape
        let mut materialized = TensorHandle::empty(client, lhs_target_shape.clone(), lhs.dtype);
        kernels::copy_reshape::<R, E>(client, &permuted, &mut materialized)?;
        materialized
    } else if lhs_needs_transpose {
        // Just permute (no reshape) - can do zero-copy
        let new_shape: Vec<usize> = lhs_perm.iter().map(|&i| lhs.shape[i]).collect();
        let new_strides: Vec<usize> = lhs_perm.iter().map(|&i| lhs.strides[i]).collect();
        let mut t = lhs.clone();
        t.shape = new_shape;
        t.strides = new_strides;
        t
    } else if lhs_needs_reshape {
        // Just reshape (no permute) - can change shape directly
        let mut t = lhs.clone();
        t.shape = lhs_target_shape.clone();
        t.strides = compute_strides(&t.shape);
        t
    } else {
        lhs.clone()
    };

    let rhs_reshaped = if rhs_needs_transpose && rhs_needs_reshape {
        // Need to materialize: copy data in permuted order, then reshape
        let permuted_shape: Vec<usize> = rhs_perm.iter().map(|&i| rhs.shape[i]).collect();
        let permuted_strides: Vec<usize> = rhs_perm.iter().map(|&i| rhs.strides[i]).collect();

        // Create permuted view
        let mut permuted = rhs.clone();
        permuted.shape = permuted_shape;
        permuted.strides = permuted_strides;

        // Materialize into contiguous tensor with target shape
        let mut materialized = TensorHandle::empty(client, rhs_target_shape.clone(), rhs.dtype);
        kernels::copy_reshape::<R, E>(client, &permuted, &mut materialized)?;
        materialized
    } else if rhs_needs_transpose {
        // Just permute (no reshape) - can do zero-copy
        let new_shape: Vec<usize> = rhs_perm.iter().map(|&i| rhs.shape[i]).collect();
        let new_strides: Vec<usize> = rhs_perm.iter().map(|&i| rhs.strides[i]).collect();
        let mut t = rhs.clone();
        t.shape = new_shape;
        t.strides = new_strides;
        t
    } else if rhs_needs_reshape {
        // Just reshape (no permute) - can change shape directly
        let mut t = rhs.clone();
        t.shape = rhs_target_shape.clone();
        t.strides = compute_strides(&t.shape);
        t
    } else {
        rhs.clone()
    };

    // Ensure output has at least 2 dimensions for matmul
    // If output is 1D (happens with some contraction patterns), reshape to 2D
    let mut output_for_matmul = output.clone();
    let original_output_shape = output.shape.clone();
    let needs_reshape = output.shape.len() < 2;

    if needs_reshape {
        // Reshape output to 2D by adding dimensions of size 1
        if !batch_shape.is_empty() {
            // Batched case: [batch..., M, N] where M or N might be 1
            let _total_elements: usize = output.shape.iter().product();
            let _batch_size: usize = batch_shape.iter().product();
            output_for_matmul.shape = [batch_shape.clone(), vec![m.max(1), n.max(1)]].concat();
        } else {
            // Non-batched case: reshape to [m, n]
            output_for_matmul.shape = vec![m, n];
        }
        output_for_matmul.strides = compute_strides(&output_for_matmul.shape);
    }

    // Perform batched GEMM: [batch..., M, K] @ [batch..., K, N] -> [batch..., M, N]
    // The cubek-matmul library will automatically detect and handle batch dimensions
    let elem_type = MatmulElemType::new(E::as_type_native_unchecked(), false);
    let dtypes = MatmulElems::from_single_dtype(elem_type);

    let lhs_handle = MatmulInputHandle::Normal(lhs_reshaped);
    let rhs_handle = MatmulInputHandle::Normal(rhs_reshaped);

    launch(
        &MatmulStrategy::Auto,
        client,
        lhs_handle,
        rhs_handle,
        output_for_matmul,
        dtypes,
    ).map_err(|e| EinsumError::launch(alloc::format!("contraction failed: {:?}", e)))?;

    // Restore original shape if we reshaped
    if needs_reshape {
        output.shape = original_output_shape;
        output.strides = compute_strides(&output.shape);
    }

    Ok(())
}

/// Checks if a permutation is the identity (no reordering needed).
fn is_identity_permutation(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

#[cfg(test)]
mod tests {
    // Integration tests would go here, but require a runtime
}
