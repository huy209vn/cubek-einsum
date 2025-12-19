//! Einsum notation parser.
//!
//! Parses strings like "ij,jk->ik" into structured EinsumNotation.

use alloc::vec::Vec;

use super::subscript::Subscript;
use super::notation::EinsumNotation;
use crate::error::{EinsumError, EinsumResult};

/// Parses an einsum notation string.
///
/// # Grammar
///
/// ```text
/// einsum      ::= inputs '->' output | inputs
/// inputs      ::= subscript (',' subscript)*
/// output      ::= subscript
/// subscript   ::= index* | '...' index* | index* '...' | index* '...' index*
/// index       ::= [a-zA-Z]
/// ```
///
/// # Examples
///
/// ```ignore
/// let notation = parse_einsum("ij,jk->ik")?;  // Matrix multiply
/// let notation = parse_einsum("...ij,...jk->...ik")?;  // Batched matmul
/// let notation = parse_einsum("ij,jk")?;  // Implicit output
/// ```
pub fn parse_einsum(notation: &str) -> EinsumResult<EinsumNotation> {
    let notation = notation.trim();

    if notation.is_empty() {
        return Err(EinsumError::parse("empty notation"));
    }

    // Split into inputs and output
    let (inputs_str, output_str) = if let Some(arrow_pos) = notation.find("->") {
        let inputs = &notation[..arrow_pos];
        let output = &notation[arrow_pos + 2..];
        (inputs, Some(output))
    } else {
        (notation, None)
    };

    // Parse input subscripts
    let input_strs: Vec<&str> = inputs_str.split(',').collect();
    if input_strs.is_empty() || (input_strs.len() == 1 && input_strs[0].is_empty()) {
        return Err(EinsumError::NoInputs);
    }

    let mut inputs = Vec::with_capacity(input_strs.len());
    for input_str in &input_strs {
        inputs.push(parse_subscript(input_str.trim())?);
    }

    // Parse or infer output subscript
    let output = if let Some(out_str) = output_str {
        parse_subscript(out_str.trim())?
    } else {
        // Implicit output: indices appearing exactly once, sorted
        infer_output(&inputs)?
    };

    let mut result = EinsumNotation::new(inputs, output);
    result = result.with_original(notation);

    Ok(result)
}

/// Parses a single subscript string into a Subscript.
fn parse_subscript(s: &str) -> EinsumResult<Subscript> {
    let mut subscript = Subscript::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            // Check for ellipsis
            '.' => {
                // Must be followed by ".."
                if chars.next() != Some('.') || chars.next() != Some('.') {
                    return Err(EinsumError::parse("incomplete ellipsis, expected '...'"));
                }
                if subscript.has_ellipsis() {
                    return Err(EinsumError::parse("multiple ellipses in subscript"));
                }
                subscript.push_ellipsis();
            }
            // Named index
            'a'..='z' | 'A'..='Z' => {
                subscript.push_named(c);
            }
            // Whitespace is ignored
            ' ' | '\t' => continue,
            // Invalid character
            _ => {
                return Err(EinsumError::parse(alloc::format!(
                    "invalid character '{}' in subscript",
                    c
                )));
            }
        }
    }

    Ok(subscript)
}

/// Infers the output subscript when not explicitly provided.
///
/// Rules (NumPy einsum convention):
/// 1. Indices appearing exactly once across all inputs appear in output
/// 2. Output indices are sorted alphabetically
/// 3. If any input has ellipsis, output has ellipsis at the beginning
fn infer_output(inputs: &[Subscript]) -> EinsumResult<Subscript> {
    use hashbrown::HashMap;

    // Count occurrences of each index
    let mut counts: HashMap<char, usize> = HashMap::new();
    let mut has_ellipsis = false;

    for input in inputs {
        if input.has_ellipsis() {
            has_ellipsis = true;
        }
        for c in input.named_indices() {
            *counts.entry(c).or_insert(0) += 1;
        }
    }

    // Collect indices that appear exactly once, sorted
    let mut output_indices: Vec<char> = counts
        .iter()
        .filter(|&(_, count)| *count == 1)
        .map(|(&c, _)| c)
        .collect();
    output_indices.sort();

    // Build output subscript
    let mut output = Subscript::new();
    if has_ellipsis {
        output.push_ellipsis();
    }
    for c in output_indices {
        output.push_named(c);
    }

    Ok(output)
}

/// Parses multiple einsum operations (for potential fusion).
///
/// Format: "ij,jk->ik; ik,kl->il"
#[allow(dead_code)]
pub fn parse_einsum_chain(notation: &str) -> EinsumResult<Vec<EinsumNotation>> {
    notation
        .split(';')
        .map(|s| parse_einsum(s.trim()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        assert_eq!(notation.num_inputs(), 2);
        assert_eq!(notation.inputs()[0].to_string(), "ij");
        assert_eq!(notation.inputs()[1].to_string(), "jk");
        assert_eq!(notation.output().to_string(), "ik");
    }

    #[test]
    fn test_parse_batched_matmul() {
        let notation = parse_einsum("bij,bjk->bik").unwrap();
        assert_eq!(notation.num_inputs(), 2);
        assert!(notation.batch_indices().contains(&'b'));
    }

    #[test]
    fn test_parse_ellipsis() {
        let notation = parse_einsum("...ij,...jk->...ik").unwrap();
        assert!(notation.has_ellipsis());
        assert!(notation.inputs()[0].has_ellipsis());
        assert!(notation.inputs()[1].has_ellipsis());
        assert!(notation.output().has_ellipsis());
    }

    #[test]
    fn test_parse_transpose() {
        let notation = parse_einsum("ij->ji").unwrap();
        assert_eq!(notation.num_inputs(), 1);
        assert!(notation.is_unary());
    }

    #[test]
    fn test_parse_trace() {
        let notation = parse_einsum("ii->").unwrap();
        assert!(notation.is_scalar_output());
        assert!(notation.contraction_indices().contains(&'i'));
    }

    #[test]
    fn test_parse_hadamard() {
        let notation = parse_einsum("ij,ij->ij").unwrap();
        assert!(notation.is_permutation_only());
    }

    #[test]
    fn test_parse_outer_product() {
        let notation = parse_einsum("i,j->ij").unwrap();
        assert!(!notation.contraction_indices().contains(&'i'));
        assert!(!notation.contraction_indices().contains(&'j'));
    }

    #[test]
    fn test_parse_reduction() {
        let notation = parse_einsum("ij->i").unwrap();
        assert!(notation.contraction_indices().contains(&'j'));
    }

    #[test]
    fn test_implicit_output() {
        // ij,jk should imply ->ik
        let notation = parse_einsum("ij,jk").unwrap();
        let output_str = notation.output().to_string();
        assert!(output_str.contains('i'));
        assert!(output_str.contains('k'));
        assert!(!output_str.contains('j'));  // j appears twice, so contracted
    }

    #[test]
    fn test_implicit_trace() {
        // ii should imply -> (empty output)
        let notation = parse_einsum("ii").unwrap();
        assert!(notation.is_scalar_output());
    }

    #[test]
    fn test_parse_dot_product() {
        let notation = parse_einsum("i,i->").unwrap();
        assert!(notation.is_scalar_output());
        assert!(notation.contraction_indices().contains(&'i'));
    }

    #[test]
    fn test_parse_attention_scores() {
        let notation = parse_einsum("bhqd,bhkd->bhqk").unwrap();
        assert_eq!(notation.num_inputs(), 2);
        assert!(notation.batch_indices().contains(&'b'));
        assert!(notation.batch_indices().contains(&'h'));
        assert!(notation.contraction_indices().contains(&'d'));
    }

    #[test]
    fn test_parse_whitespace() {
        let notation = parse_einsum(" ij , jk -> ik ").unwrap();
        assert_eq!(notation.num_inputs(), 2);
    }

    #[test]
    fn test_parse_error_invalid_char() {
        let result = parse_einsum("i1j,jk->ik");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_incomplete_ellipsis() {
        let result = parse_einsum("..ij,jk->ik");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_empty() {
        let result = parse_einsum("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_three_inputs() {
        let notation = parse_einsum("ijk,jkl,klm->im").unwrap();
        assert_eq!(notation.num_inputs(), 3);
    }

    #[test]
    fn test_parse_uppercase() {
        let notation = parse_einsum("IJ,JK->IK").unwrap();
        assert!(notation.contraction_indices().contains(&'J'));
    }
}
