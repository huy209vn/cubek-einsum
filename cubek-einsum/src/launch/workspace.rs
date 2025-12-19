use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use hashbrown::{HashSet, HashMap};

use cubecl::Runtime;
use cubecl::client::ComputeClient;
use cubecl::ir::StorageType;
use cubecl::std::tensor::TensorHandle;

use crate::error::{EinsumError, EinsumResult};

/// Workspace manager for einsum operations.
///
/// Manages temporary tensor allocations for multi-step operations,
/// with memory reuse and efficient allocation strategies.
pub struct Workspace<R: Runtime> {
    /// Cached tensor handles for reuse.
    tensors: Vec<TensorHandle<R>>,

    /// Tracks which tensors are currently in use (by step index)
    tensor_usage: BTreeMap<usize, HashSet<usize>>,

    /// Maximum total memory to allocate (0 = unlimited)
    max_size: usize,
}

/// Information about a tensor allocation.
#[derive(Debug)]
#[allow(dead_code)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: StorageType,
    pub size_bytes: usize,
}

impl<R: Runtime> Workspace<R> {
    /// Creates a new empty workspace with optional memory limit.
    pub fn new(max_size: usize) -> Self {
        Self {
            tensors: Vec::new(),
            tensor_usage: BTreeMap::new(),
            max_size,
        }
    }

    /// Registers input tensors that can potentially be reused.
    ///
    /// Input tensors may be used directly in contractions if they're
    /// not needed after their first use.
    pub fn register_inputs(&mut self, num_inputs: usize) {
        // Mark all inputs as used at step 0
        let mut usage = HashSet::new();
        for i in 0..num_inputs {
            usage.insert(i);
        }
        self.tensor_usage.insert(0, usage);
    }

    /// Marks a tensor as used at a specific step.
    pub fn mark_used(&mut self, tensor_idx: usize, step_idx: usize) {
        self.tensor_usage.entry(step_idx).or_default().insert(tensor_idx);
    }

    /// Allocates a new temporary tensor with the given shape.
    ///
    /// The tensor is initialized to zeros. Returns the index of the allocation.
    pub fn alloc(
        &mut self,
        client: &ComputeClient<R>,
        shape: Vec<usize>,
        dtype: StorageType,
        step_idx: usize,
    ) -> EinsumResult<usize> {
        let size_bytes = compute_tensor_size(&shape, dtype)?;

        // Check memory limit
        if self.max_size > 0 {
            let current_usage = self.current_memory_usage()?;
            if current_usage + size_bytes > self.max_size {
                return Err(EinsumError::memory(alloc::format!(
                    "workspace limit exceeded: {} > {}",
                    current_usage + size_bytes,
                    self.max_size
                )));
            }
        }

        // Try to find reusable allocation
        if let Some(reused_idx) = self.find_reusable_allocation(size_bytes, &shape) {
            return Ok(reused_idx);
        }

        // Allocate new tensor
        let tensor = TensorHandle::zeros(client, shape.clone(), dtype);

        // Track usage
        self.tensor_usage.entry(step_idx).or_default().insert(self.tensors.len());
        self.tensors.push(tensor);

        Ok(self.tensors.len() - 1)
    }

    /// Gets a reference to an allocated tensor.
    pub fn get(&self, idx: usize) -> Option<&TensorHandle<R>> {
        self.tensors.get(idx)
    }

    /// Gets a mutable reference to an allocated tensor.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut TensorHandle<R>> {
        self.tensors.get_mut(idx)
    }

    /// Finds tensors that can be freed after the given step.
    pub fn find_freed_after_step(&self, step_idx: usize) -> Vec<usize> {
        let mut freed = Vec::new();

        for (usage_step, tensor_indices) in &self.tensor_usage {
            if *usage_step <= step_idx {
                for &tensor_idx in tensor_indices {
                    // Check if this tensor is used in any later step
                    let mut still_used = false;
                    for (&later_step, indices) in &self.tensor_usage {
                        if later_step > step_idx && indices.contains(&tensor_idx) {
                            still_used = true;
                            break;
                        }
                    }

                    if !still_used {
                        freed.push(tensor_idx);
                    }
                }
            }
        }

        // Deduplicate
        freed.sort();
        freed.dedup();
        freed
    }

    /// Finds a reusable allocation of sufficient size and compatible shape.
    fn find_reusable_allocation(
        &mut self,
        min_size: usize,
        target_shape: &[usize],
    ) -> Option<usize> {
        // Simple strategy: return first allocation that's large enough
        for (i, tensor) in self.tensors.iter().enumerate() {
            let size = compute_tensor_size(tensor.shape.as_slice(), tensor.dtype).ok()?;
            if size >= min_size && can_reuse_shape(&tensor.shape, target_shape) {
                // Move to end and return index
                let tensor = self.tensors.remove(i);
                self.tensors.push(tensor);
                return Some(self.tensors.len() - 1);
            }
        }
        None
    }

    /// Computes current memory usage.
    fn current_memory_usage(&self) -> EinsumResult<usize> {
        let mut total = 0;
        for tensor in &self.tensors {
            total += compute_tensor_size(tensor.shape.as_slice(), tensor.dtype)?;
        }
        Ok(total)
    }

    /// Returns the number of allocated tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Checks if the workspace is empty.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Clears all temporary allocations.
    pub fn clear(&mut self) {
        self.tensors.clear();
        self.tensor_usage.clear();
    }
}

impl<R: Runtime> Default for Workspace<R> {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Computes the size in bytes of a tensor.
fn compute_tensor_size(shape: &[usize], dtype: StorageType) -> EinsumResult<usize> {
    let elem_size = dtype.size();
    let num_elements = shape.iter().product::<usize>();
    Ok(num_elements * elem_size)
}

/// Checks if one shape can reuse another's memory.
///
/// Two shapes are compatible if:
/// 1. They have the same number of dimensions
/// 2. The product of dimensions is the same (same total elements)
fn can_reuse_shape(existing: &[usize], target: &[usize]) -> bool {
    if existing.len() != target.len() {
        return false;
    }

    let existing_size = existing.iter().product::<usize>();
    let target_size = target.iter().product::<usize>();

    existing_size == target_size
}

/// Computes the output shape for a contraction step.
///
/// Given two input tensors with their index labels and the result indices,
/// computes what the output tensor shape should be.
#[allow(dead_code)]
pub fn compute_contraction_output_shape(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    lhs_indices: &[char],
    rhs_indices: &[char],
    result_indices: &[char],
) -> EinsumResult<Vec<usize>> {
    // Build mapping from index to dimension size
    let mut dim_map = HashMap::new();

    // Add dimensions from left tensor
    for (&idx_char, &size) in lhs_indices.iter().zip(lhs_shape.iter()) {
        dim_map.insert(idx_char, size);
    }

    // Add dimensions from right tensor
    for (&idx_char, &size) in rhs_indices.iter().zip(rhs_shape.iter()) {
        dim_map.insert(idx_char, size);
    }

    // Build output shape in order of result indices
    let mut output_shape = Vec::with_capacity(result_indices.len());
    for &idx_char in result_indices {
        if let Some(&size) = dim_map.get(&idx_char) {
            output_shape.push(size);
        } else {
            return Err(EinsumError::shape(
                alloc::format!("missing dimension for index '{}'", idx_char)
            ));
        }
    }

    Ok(output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_compatibility() {
        assert!(can_reuse_shape(&[10, 20], &[10, 20]));
        assert!(can_reuse_shape(&[5, 40], &[10, 20])); // Same total elements
        assert!(!can_reuse_shape(&[10, 20], &[10]));   // Different rank
    }

    #[test]
    fn test_output_shape_computation() {
        // Test: ij,jk->ik where i=10,j=20,k=30
        let lhs_shape = [10, 20];
        let rhs_shape = [20, 30];
        let lhs_indices = ['i', 'j'];
        let rhs_indices = ['j', 'k'];
        let result = ['i', 'k'];

        let output = compute_contraction_output_shape(
            &lhs_shape,
            &rhs_shape,
            &lhs_indices,
            &rhs_indices,
            &result
        ).unwrap();

        assert_eq!(output, vec![10, 30]);
    }

    #[test]
    fn test_output_shape_computation_batched() {
        // Test: bij,bjk->bik where b=5,i=10,j=20,k=30
        let lhs_shape = [5, 10, 20];
        let rhs_shape = [5, 20, 30];
        let lhs_indices = ['b', 'i', 'j'];
        let rhs_indices = ['b', 'j', 'k'];
        let result = ['b', 'i', 'k'];

        let output = compute_contraction_output_shape(
            &lhs_shape,
            &rhs_shape,
            &lhs_indices,
            &rhs_indices,
            &result
        ).unwrap();

        assert_eq!(output, vec![5, 10, 30]);
    }
}
