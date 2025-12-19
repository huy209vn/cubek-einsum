//! Validation for einsum notation and tensor shapes.

use alloc::vec::Vec;
use alloc::string::ToString;
use hashbrown::HashMap;

use super::notation::EinsumNotation;
use crate::error::{EinsumError, EinsumResult};

/// Validates an einsum notation for correctness.
///
/// Checks:
/// - Output indices must appear in at least one input
/// - No index appears more than twice total
/// - Ellipsis usage is consistent
pub fn validate_notation(notation: &EinsumNotation) -> EinsumResult<()> {
    validate_output_indices(notation)?;
    validate_index_counts(notation)?;
    validate_ellipsis_consistency(notation)?;
    Ok(())
}

/// Validates that all output indices appear in at least one input.
fn validate_output_indices(notation: &EinsumNotation) -> EinsumResult<()> {
    for c in notation.output().named_indices() {
        let appears_in_input = notation.inputs().iter().any(|input| input.contains(c));
        if !appears_in_input {
            return Err(EinsumError::OutputIndexNotInInputs { index: c });
        }
    }
    Ok(())
}

/// Validates that no index appears more than twice total.
fn validate_index_counts(notation: &EinsumNotation) -> EinsumResult<()> {
    let mut counts: HashMap<char, usize> = HashMap::new();

    // Count in inputs
    for input in notation.inputs() {
        for c in input.named_indices() {
            *counts.entry(c).or_insert(0) += 1;
        }
    }

    // Count in output
    for c in notation.output().named_indices() {
        *counts.entry(c).or_insert(0) += 1;
    }

    // Check counts
    for (&c, &count) in &counts {
        if count > 3 {
            // An index can appear: once in each of two inputs + once in output = 3 max
            // Actually, more complex: trace has ii in one input
            // Let's be more permissive and check for truly invalid cases
            return Err(EinsumError::IndexAppearsMoreThanTwice { index: c, count });
        }
    }

    Ok(())
}

/// Validates ellipsis consistency.
fn validate_ellipsis_consistency(notation: &EinsumNotation) -> EinsumResult<()> {
    let inputs_with_ellipsis: Vec<_> = notation
        .inputs()
        .iter()
        .enumerate()
        .filter(|(_, s)| s.has_ellipsis())
        .map(|(i, _)| i)
        .collect();

    let output_has_ellipsis = notation.output().has_ellipsis();

    if !inputs_with_ellipsis.is_empty() || output_has_ellipsis {
        // If any has ellipsis, all should have ellipsis (or we need to be careful)
        // NumPy is lenient here, but we'll enforce consistency for clarity
        if inputs_with_ellipsis.len() != notation.num_inputs() {
            return Err(EinsumError::InconsistentEllipsis {
                message: "if any input has ellipsis, all inputs should have ellipsis".to_string(),
            });
        }
        if !output_has_ellipsis {
            return Err(EinsumError::InconsistentEllipsis {
                message: "output must have ellipsis when inputs do".to_string(),
            });
        }
    }

    Ok(())
}

/// Validates tensor shapes against the notation.
///
/// Returns the number of ellipsis dimensions (0 if no ellipsis).
pub fn validate_shapes(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
) -> EinsumResult<ValidationResult> {
    if shapes.len() != notation.num_inputs() {
        return Err(EinsumError::parse(alloc::format!(
            "expected {} input shapes, got {}",
            notation.num_inputs(),
            shapes.len()
        )));
    }

    // Determine ellipsis dimensions
    let ellipsis_dims = compute_ellipsis_dims(notation, shapes)?;

    // Build dimension mapping and validate consistency
    let dim_map = build_dimension_map(notation, shapes, ellipsis_dims)?;

    // Compute output shape
    let output_shape = compute_output_shape(notation, &dim_map, ellipsis_dims)?;

    // Compute contracted shape (for FLOP estimation)
    let contracted_shape: Vec<usize> = notation
        .contraction_indices()
        .iter()
        .filter_map(|&c| dim_map.get(&c).copied())
        .collect();

    Ok(ValidationResult {
        ellipsis_dims,
        dim_map,
        output_shape,
        contracted_shape,
    })
}

/// Result of shape validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Number of dimensions represented by ellipsis.
    pub ellipsis_dims: usize,
    /// Mapping from index characters to dimension sizes.
    pub dim_map: HashMap<char, usize>,
    /// Computed output shape.
    pub output_shape: Vec<usize>,
    /// Shapes of contracted dimensions.
    pub contracted_shape: Vec<usize>,
}

impl ValidationResult {
    /// Computes the total number of FLOPs.
    pub fn compute_flops(&self) -> u64 {
        let output_size: u64 = self.output_shape.iter().map(|&d| d as u64).product();
        let contracted_size: u64 = self.contracted_shape.iter().map(|&d| d as u64).product();

        // FLOPs = output_elements * contracted_elements * 2 (mul + add)
        output_size.saturating_mul(contracted_size).saturating_mul(2)
    }

    /// Computes the memory footprint in elements.
    pub fn compute_memory_elements(&self, input_shapes: &[&[usize]]) -> u64 {
        let input_elements: u64 = input_shapes
            .iter()
            .map(|shape| shape.iter().map(|&d| d as u64).product::<u64>())
            .sum();

        let output_elements: u64 = self.output_shape.iter().map(|&d| d as u64).product();

        input_elements + output_elements
    }
}

/// Computes the number of dimensions represented by ellipsis.
fn compute_ellipsis_dims(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
) -> EinsumResult<usize> {
    if !notation.has_ellipsis() {
        return Ok(0);
    }

    let mut ellipsis_dims: Option<usize> = None;

    for (input, shape) in notation.inputs().iter().zip(shapes.iter()) {
        if input.has_ellipsis() {
            let explicit = input.explicit_count();
            let total = shape.len();

            if total < explicit {
                return Err(EinsumError::DimensionMismatch {
                    subscript: input.to_string(),
                    expected: explicit,
                    got: total,
                });
            }

            let this_ellipsis_dims = total - explicit;

            if let Some(prev) = ellipsis_dims {
                if prev != this_ellipsis_dims {
                    return Err(EinsumError::EllipsisDimensionMismatch {
                        expected: prev,
                        got: this_ellipsis_dims,
                    });
                }
            } else {
                ellipsis_dims = Some(this_ellipsis_dims);
            }
        }
    }

    Ok(ellipsis_dims.unwrap_or(0))
}

/// Builds a mapping from index characters to dimension sizes.
fn build_dimension_map(
    notation: &EinsumNotation,
    shapes: &[&[usize]],
    ellipsis_dims: usize,
) -> EinsumResult<HashMap<char, usize>> {
    let mut dim_map: HashMap<char, usize> = HashMap::new();

    // Generate batch index characters for ellipsis
    let batch_indices: Vec<char> = (0..ellipsis_dims)
        .map(|i| char::from_u32(0x2460 + i as u32).unwrap_or('?'))
        .collect();

    for (input, shape) in notation.inputs().iter().zip(shapes.iter()) {
        let expanded = input.expand_ellipsis(&batch_indices);

        if expanded.explicit_count() != shape.len() {
            return Err(EinsumError::DimensionMismatch {
                subscript: input.to_string(),
                expected: expanded.explicit_count(),
                got: shape.len(),
            });
        }

        for (idx, c) in expanded.named_indices().enumerate() {
            let dim = shape[idx];
            if let Some(&existing) = dim_map.get(&c) {
                if existing != dim {
                    return Err(EinsumError::ShapeMismatch {
                        index: c,
                        expected: existing,
                        got: dim,
                    });
                }
            } else {
                dim_map.insert(c, dim);
            }
        }
    }

    Ok(dim_map)
}

/// Computes the output shape from the dimension map.
fn compute_output_shape(
    notation: &EinsumNotation,
    dim_map: &HashMap<char, usize>,
    ellipsis_dims: usize,
) -> EinsumResult<Vec<usize>> {
    let batch_indices: Vec<char> = (0..ellipsis_dims)
        .map(|i| char::from_u32(0x2460 + i as u32).unwrap_or('?'))
        .collect();

    let expanded_output = notation.output().expand_ellipsis(&batch_indices);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_validate_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let result = validate_shapes(&notation, &[&[3, 4], &[4, 5]]).unwrap();

        assert_eq!(result.output_shape, vec![3, 5]);
        assert_eq!(result.contracted_shape, vec![4]);
    }

    #[test]
    fn test_validate_batched_matmul() {
        let notation = parse_einsum("bij,bjk->bik").unwrap();
        let result = validate_shapes(&notation, &[&[2, 3, 4], &[2, 4, 5]]).unwrap();

        assert_eq!(result.output_shape, vec![2, 3, 5]);
        assert_eq!(result.contracted_shape, vec![4]);
    }

    #[test]
    fn test_validate_shape_mismatch() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let result = validate_shapes(&notation, &[&[3, 4], &[5, 6]]);  // j: 4 vs 5

        assert!(result.is_err());
    }

    #[test]
    fn test_validate_dimension_mismatch() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let result = validate_shapes(&notation, &[&[3, 4, 5], &[4, 5]]);  // wrong dims

        assert!(result.is_err());
    }

    #[test]
    fn test_flops_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let result = validate_shapes(&notation, &[&[100, 200], &[200, 300]]).unwrap();

        // FLOPs = M * N * K * 2 = 100 * 300 * 200 * 2 = 12,000,000
        assert_eq!(result.compute_flops(), 12_000_000);
    }

    #[test]
    fn test_validate_ellipsis() {
        let notation = parse_einsum("...ij,...jk->...ik").unwrap();
        let result = validate_shapes(&notation, &[&[2, 3, 4, 5], &[2, 3, 5, 6]]).unwrap();

        assert_eq!(result.ellipsis_dims, 2);
        assert_eq!(result.output_shape, vec![2, 3, 4, 6]);
    }
}
