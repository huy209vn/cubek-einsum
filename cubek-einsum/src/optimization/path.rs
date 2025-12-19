//! Contraction path representation.

use alloc::vec::Vec;

/// A single step in a contraction path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionStep {
    /// Indices of tensors to contract (in current tensor list).
    pub inputs: (usize, usize),
    /// Indices that are contracted (summed) in this step.
    pub contracted_indices: Vec<char>,
    /// Indices in the result tensor.
    pub result_indices: Vec<char>,
    /// Estimated cost of this step.
    pub estimated_flops: u64,
}

impl ContractionStep {
    pub fn new(
        inputs: (usize, usize),
        contracted_indices: Vec<char>,
        result_indices: Vec<char>,
        estimated_flops: u64,
    ) -> Self {
        Self {
            inputs,
            contracted_indices,
            result_indices,
            estimated_flops,
        }
    }
}

/// A complete contraction path.
#[derive(Debug, Clone)]
pub struct ContractionPath {
    /// Steps to execute in order.
    steps: Vec<ContractionStep>,
    /// Total estimated FLOPs.
    total_flops: u64,
    /// Total estimated memory traffic.
    total_memory: u64,
}

impl ContractionPath {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            total_flops: 0,
            total_memory: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            steps: Vec::with_capacity(capacity),
            total_flops: 0,
            total_memory: 0,
        }
    }

    pub fn push(&mut self, step: ContractionStep) {
        self.total_flops = self.total_flops.saturating_add(step.estimated_flops);
        self.steps.push(step);
    }

    pub fn steps(&self) -> &[ContractionStep] {
        &self.steps
    }

    pub fn total_flops(&self) -> u64 {
        self.total_flops
    }

    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Converts path to a list of (i, j) pairs indicating which tensors to contract.
    pub fn to_pairs(&self) -> Vec<(usize, usize)> {
        self.steps.iter().map(|s| s.inputs).collect()
    }
}

impl Default for ContractionPath {
    fn default() -> Self {
        Self::new()
    }
}

/// Intermediate state during path search.
#[derive(Debug, Clone)]
pub struct TensorState {
    /// Current tensor shapes.
    pub shapes: Vec<Vec<usize>>,
    /// Current tensor indices.
    pub indices: Vec<Vec<char>>,
    /// Mapping from original tensor index to current position.
    pub original_indices: Vec<usize>,
}

impl TensorState {
    pub fn new(shapes: Vec<Vec<usize>>, indices: Vec<Vec<char>>) -> Self {
        let original_indices = (0..shapes.len()).collect();
        Self {
            shapes,
            indices,
            original_indices,
        }
    }

    /// Number of tensors remaining.
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Contract tensors at positions i and j, returning the new state.
    pub fn contract(&self, i: usize, j: usize, output_indices: &[char]) -> TensorState {
        assert!(i < j && j < self.len());

        let mut new_shapes = Vec::with_capacity(self.len() - 1);
        let mut new_indices = Vec::with_capacity(self.len() - 1);
        let mut new_original = Vec::with_capacity(self.len() - 1);

        // Compute result shape
        let mut dim_map: hashbrown::HashMap<char, usize> = hashbrown::HashMap::new();
        for (&c, &d) in self.indices[i].iter().zip(self.shapes[i].iter()) {
            dim_map.insert(c, d);
        }
        for (&c, &d) in self.indices[j].iter().zip(self.shapes[j].iter()) {
            dim_map.insert(c, d);
        }

        let result_shape: Vec<usize> = output_indices
            .iter()
            .map(|c| dim_map.get(c).copied().unwrap_or(1))
            .collect();

        // Build new tensor list
        for k in 0..self.len() {
            if k != i && k != j {
                new_shapes.push(self.shapes[k].clone());
                new_indices.push(self.indices[k].clone());
                new_original.push(self.original_indices[k]);
            }
        }

        // Add result tensor at the end
        new_shapes.push(result_shape);
        new_indices.push(output_indices.to_vec());
        // Use smaller original index for result
        new_original.push(self.original_indices[i].min(self.original_indices[j]));

        TensorState {
            shapes: new_shapes,
            indices: new_indices,
            original_indices: new_original,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contraction_path() {
        let mut path = ContractionPath::new();

        path.push(ContractionStep::new(
            (0, 1),
            vec!['j'],
            vec!['i', 'k'],
            1000,
        ));

        assert_eq!(path.len(), 1);
        assert_eq!(path.total_flops(), 1000);
    }

    #[test]
    fn test_tensor_state_contract() {
        let state = TensorState::new(
            vec![vec![3, 4], vec![4, 5], vec![5, 6]],
            vec![vec!['i', 'j'], vec!['j', 'k'], vec!['k', 'l']],
        );

        // Contract tensors 0 and 1 (ij,jk->ik)
        let new_state = state.contract(0, 1, &['i', 'k']);

        assert_eq!(new_state.len(), 2);
        assert_eq!(new_state.shapes[0], vec![5, 6]); // tensor 2, now at position 0
        assert_eq!(new_state.shapes[1], vec![3, 5]); // result of 0*1
    }
}
