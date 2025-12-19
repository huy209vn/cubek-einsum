//! Complete einsum notation representation.

use alloc::vec;
use alloc::vec::Vec;
use alloc::collections::BTreeSet;
use alloc::string::String;
use core::fmt;

use super::subscript::Subscript;

/// Complete parsed einsum notation.
///
/// Contains the input subscripts, output subscript, and derived index information.
#[derive(Debug, Clone)]
pub struct EinsumNotation {
    /// Input tensor subscripts.
    inputs: Vec<Subscript>,
    /// Output tensor subscript.
    output: Subscript,
    /// Indices that appear in inputs but not output (contracted/summed).
    contraction_indices: BTreeSet<char>,
    /// Indices that appear in all inputs and output (batch dimensions).
    batch_indices: BTreeSet<char>,
    /// Indices that appear in output (free indices, ordered).
    output_indices: Vec<char>,
    /// Whether any subscript uses ellipsis.
    has_ellipsis: bool,
    /// Original notation string (if available).
    original: Option<String>,
}

impl EinsumNotation {
    /// Creates a new einsum notation from parsed components.
    pub fn new(inputs: Vec<Subscript>, output: Subscript) -> Self {
        let has_ellipsis = inputs.iter().any(|s| s.has_ellipsis()) || output.has_ellipsis();

        // Collect all input indices with their counts
        let mut all_input_indices: BTreeSet<char> = BTreeSet::new();
        let mut input_index_counts: hashbrown::HashMap<char, usize> = hashbrown::HashMap::new();

        for input in &inputs {
            for c in input.named_indices() {
                all_input_indices.insert(c);
                *input_index_counts.entry(c).or_insert(0) += 1;
            }
        }

        // Output indices (in order)
        let output_indices: Vec<char> = output.named_indices().collect();
        let output_set: BTreeSet<char> = output_indices.iter().copied().collect();

        // Contraction indices: appear in inputs but not in output
        let contraction_indices: BTreeSet<char> = all_input_indices
            .difference(&output_set)
            .copied()
            .collect();

        // Batch indices: appear in all inputs AND in output
        // (indices that are preserved across all tensors)
        let mut batch_indices = BTreeSet::new();
        for &c in &output_set {
            let appears_in_all = inputs.iter().all(|input| input.contains(c));
            if appears_in_all {
                batch_indices.insert(c);
            }
        }

        Self {
            inputs,
            output,
            contraction_indices,
            batch_indices,
            output_indices,
            has_ellipsis,
            original: None,
        }
    }

    /// Sets the original notation string.
    pub fn with_original(mut self, original: impl Into<String>) -> Self {
        self.original = Some(original.into());
        self
    }

    /// Returns the input subscripts.
    #[inline]
    pub fn inputs(&self) -> &[Subscript] {
        &self.inputs
    }

    /// Returns the output subscript.
    #[inline]
    pub fn output(&self) -> &Subscript {
        &self.output
    }

    /// Returns the number of input tensors.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the indices that will be contracted (summed over).
    #[inline]
    pub fn contraction_indices(&self) -> &BTreeSet<char> {
        &self.contraction_indices
    }

    /// Returns the batch indices (preserved across all tensors).
    #[inline]
    pub fn batch_indices(&self) -> &BTreeSet<char> {
        &self.batch_indices
    }

    /// Returns the output indices in order.
    #[inline]
    pub fn output_indices(&self) -> &[char] {
        &self.output_indices
    }

    /// Returns true if any subscript uses ellipsis.
    #[inline]
    pub fn has_ellipsis(&self) -> bool {
        self.has_ellipsis
    }

    /// Returns true if this is a unary operation (single input).
    #[inline]
    pub fn is_unary(&self) -> bool {
        self.inputs.len() == 1
    }

    /// Returns true if this is a binary operation (two inputs).
    #[inline]
    pub fn is_binary(&self) -> bool {
        self.inputs.len() == 2
    }

    /// Returns true if this has no contractions (element-wise or permutation only).
    #[inline]
    pub fn is_permutation_only(&self) -> bool {
        self.contraction_indices.is_empty()
    }

    /// Returns true if output is scalar (empty subscript).
    #[inline]
    pub fn is_scalar_output(&self) -> bool {
        self.output.is_empty() || (self.output.len() == 1 && self.output.has_ellipsis())
    }

    /// Returns all unique indices across all inputs and output.
    pub fn all_indices(&self) -> BTreeSet<char> {
        let mut all = BTreeSet::new();
        for input in &self.inputs {
            for c in input.named_indices() {
                all.insert(c);
            }
        }
        for c in self.output.named_indices() {
            all.insert(c);
        }
        all
    }

    /// Counts total occurrences of an index across all inputs.
    pub fn count_in_inputs(&self, c: char) -> usize {
        self.inputs.iter().map(|s| s.count(c)).sum()
    }

    /// Returns which inputs contain a given index.
    pub fn inputs_containing(&self, c: char) -> Vec<usize> {
        self.inputs
            .iter()
            .enumerate()
            .filter_map(|(i, s)| if s.contains(c) { Some(i) } else { None })
            .collect()
    }

    /// Returns true if an index is contracted (summed over).
    #[inline]
    pub fn is_contracted(&self, c: char) -> bool {
        self.contraction_indices.contains(&c)
    }

    /// Returns true if an index is a batch index.
    #[inline]
    pub fn is_batch(&self, c: char) -> bool {
        self.batch_indices.contains(&c)
    }

    /// Creates a notation for a pairwise contraction of two inputs.
    ///
    /// Used during path optimization to create intermediate notation.
    pub fn pairwise(&self, input_a: usize, input_b: usize) -> EinsumNotation {
        let sub_a = &self.inputs[input_a];
        let sub_b = &self.inputs[input_b];

        // Find contracted indices between these two
        let indices_a: BTreeSet<char> = sub_a.named_indices().collect();
        let indices_b: BTreeSet<char> = sub_b.named_indices().collect();

        let common: BTreeSet<char> = indices_a.intersection(&indices_b).copied().collect();

        // Indices that appear elsewhere are kept in output
        let mut kept_elsewhere = BTreeSet::new();
        for (i, input) in self.inputs.iter().enumerate() {
            if i != input_a && i != input_b {
                for c in input.named_indices() {
                    kept_elsewhere.insert(c);
                }
            }
        }
        for c in self.output.named_indices() {
            kept_elsewhere.insert(c);
        }

        // Output indices: union of a and b, minus those only in common and not kept elsewhere
        let mut output_chars = Vec::new();
        let mut seen = BTreeSet::new();

        // Add indices from a (in order)
        for c in sub_a.named_indices() {
            if !seen.contains(&c) {
                // Keep if: in output, or appears elsewhere, or not in common
                if kept_elsewhere.contains(&c) || !common.contains(&c) {
                    output_chars.push(c);
                    seen.insert(c);
                } else if common.contains(&c) && kept_elsewhere.contains(&c) {
                    output_chars.push(c);
                    seen.insert(c);
                } else if !common.contains(&c) {
                    output_chars.push(c);
                    seen.insert(c);
                }
            }
        }

        // Add indices from b that weren't in a
        for c in sub_b.named_indices() {
            if !seen.contains(&c) {
                if kept_elsewhere.contains(&c) || !common.contains(&c) {
                    output_chars.push(c);
                    seen.insert(c);
                }
            }
        }

        // Actually, let's simplify: contracted = common indices not in final output
        // For pairwise, output = (a ∪ b) - (common - kept_elsewhere)
        output_chars.clear();
        seen.clear();

        for c in sub_a.named_indices() {
            if !seen.contains(&c) {
                let is_internal_contraction = common.contains(&c) && !kept_elsewhere.contains(&c);
                if !is_internal_contraction {
                    output_chars.push(c);
                }
                seen.insert(c);
            }
        }

        for c in sub_b.named_indices() {
            if !seen.contains(&c) {
                let is_internal_contraction = common.contains(&c) && !kept_elsewhere.contains(&c);
                if !is_internal_contraction {
                    output_chars.push(c);
                }
                seen.insert(c);
            }
        }

        let pairwise_output = Subscript::from_chars(output_chars);

        EinsumNotation::new(
            vec![sub_a.clone(), sub_b.clone()],
            pairwise_output,
        )
    }

    /// Returns the dimensions for the output tensor given input dimensions.
    pub fn compute_output_shape(
        &self,
        input_shapes: &[&[usize]],
        ellipsis_dims: usize,
    ) -> Result<Vec<usize>, super::super::error::EinsumError> {
        use super::super::error::EinsumError;

        // Build dimension mapping from indices to sizes
        let mut dim_map: hashbrown::HashMap<char, usize> = hashbrown::HashMap::new();

        for (_input_idx, (subscript, shape)) in self.inputs.iter().zip(input_shapes.iter()).enumerate() {
            let expanded = subscript.expand_ellipsis(
                &generate_batch_indices(ellipsis_dims)
            );

            if expanded.explicit_count() != shape.len() {
                return Err(EinsumError::DimensionMismatch {
                    subscript: subscript.to_string(),
                    expected: expanded.explicit_count(),
                    got: shape.len(),
                });
            }

            for (idx, c) in expanded.named_indices().enumerate() {
                if let Some(&existing) = dim_map.get(&c) {
                    if existing != shape[idx] {
                        return Err(EinsumError::ShapeMismatch {
                            index: c,
                            expected: existing,
                            got: shape[idx],
                        });
                    }
                } else {
                    dim_map.insert(c, shape[idx]);
                }
            }
        }

        // Compute output shape
        let expanded_output = self.output.expand_ellipsis(
            &generate_batch_indices(ellipsis_dims)
        );

        let mut output_shape = Vec::with_capacity(expanded_output.explicit_count());
        for c in expanded_output.named_indices() {
            if let Some(&dim) = dim_map.get(&c) {
                output_shape.push(dim);
            } else {
                return Err(EinsumError::OutputIndexNotInInputs { index: c });
            }
        }

        Ok(output_shape)
    }

    /// Computes the number of FLOPs for this einsum operation.
    pub fn compute_flops(&self, input_shapes: &[&[usize]], ellipsis_dims: usize) -> u64 {
        // Build dimension mapping
        let mut dim_map: hashbrown::HashMap<char, usize> = hashbrown::HashMap::new();
        let batch_indices = generate_batch_indices(ellipsis_dims);

        for (subscript, shape) in self.inputs.iter().zip(input_shapes.iter()) {
            let expanded = subscript.expand_ellipsis(&batch_indices);
            for (idx, c) in expanded.named_indices().enumerate() {
                dim_map.insert(c, shape.get(idx).copied().unwrap_or(1));
            }
        }

        // FLOPs = product of all dimensions (output + contracted) * 2 (mul + add)
        let mut total_product: u64 = 1;

        // Output dimensions
        let expanded_output = self.output.expand_ellipsis(&batch_indices);
        for c in expanded_output.named_indices() {
            if let Some(&d) = dim_map.get(&c) {
                total_product = total_product.saturating_mul(d as u64);
            }
        }

        // Contracted dimensions
        for &c in &self.contraction_indices {
            if let Some(&d) = dim_map.get(&c) {
                total_product = total_product.saturating_mul(d as u64);
            }
        }

        // Multiply by 2 for fused multiply-add
        total_product.saturating_mul(2)
    }
}

/// Generates batch index characters for ellipsis expansion.
fn generate_batch_indices(count: usize) -> Vec<char> {
    // Use uppercase letters starting from 'A' for batch indices
    // These won't conflict with typical lowercase indices
    (0..count)
        .map(|i| {
            // Use characters that are unlikely to conflict
            // Starting from some high Unicode point or using a pattern
            char::from_u32(0x2460 + i as u32).unwrap_or('?') // Circled digits ①②③...
        })
        .collect()
}

impl fmt::Display for EinsumNotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, input) in self.inputs.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", input)?;
        }
        write!(f, "->{}", self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_notation() {
        let a = Subscript::from_chars(['i', 'j']);
        let b = Subscript::from_chars(['j', 'k']);
        let c = Subscript::from_chars(['i', 'k']);

        let notation = EinsumNotation::new(vec![a, b], c);

        assert_eq!(notation.num_inputs(), 2);
        assert!(notation.is_binary());
        assert!(!notation.is_permutation_only());
        assert!(notation.contraction_indices().contains(&'j'));
        assert!(!notation.contraction_indices().contains(&'i'));
        assert!(!notation.contraction_indices().contains(&'k'));
    }

    #[test]
    fn test_trace_notation() {
        let a = Subscript::from_chars(['i', 'i']);
        let out = Subscript::new();

        let notation = EinsumNotation::new(vec![a], out);

        assert!(notation.is_unary());
        assert!(notation.is_scalar_output());
        assert!(notation.contraction_indices().contains(&'i'));
    }

    #[test]
    fn test_transpose_notation() {
        let a = Subscript::from_chars(['i', 'j']);
        let out = Subscript::from_chars(['j', 'i']);

        let notation = EinsumNotation::new(vec![a], out);

        assert!(notation.is_unary());
        assert!(notation.is_permutation_only());
        assert!(notation.contraction_indices().is_empty());
    }

    #[test]
    fn test_display() {
        let a = Subscript::from_chars(['i', 'j']);
        let b = Subscript::from_chars(['j', 'k']);
        let c = Subscript::from_chars(['i', 'k']);

        let notation = EinsumNotation::new(vec![a, b], c);

        assert_eq!(format!("{}", notation), "ij,jk->ik");
    }
}
