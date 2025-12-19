//! Matrix multiplication pattern detection.

use alloc::vec::Vec;
use alloc::collections::BTreeSet;

use crate::notation::EinsumNotation;

/// Configuration for a detected matmul operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatmulConfig {
    /// Whether first input needs transposition.
    pub transpose_a: bool,
    /// Whether second input needs transposition.
    pub transpose_b: bool,
    /// Batch dimension indices (empty for non-batched).
    pub batch_dims: Vec<usize>,
    /// Index of the M dimension in output.
    pub m_dim: usize,
    /// Index of the N dimension in output.
    pub n_dim: usize,
    /// The contracted (K) index character.
    pub k_index: char,
}

/// Checks if the notation represents a standard matrix multiplication.
///
/// Standard matmul: `ij,jk->ik` or variants with different index names.
/// Also detects transposed variants:
/// - `ji,jk->ik` (A transposed)
/// - `ij,kj->ik` (B transposed)
/// - `ji,kj->ik` (both transposed)
pub fn is_matmul(notation: &EinsumNotation) -> Option<MatmulConfig> {
    if !notation.is_binary() {
        return None;
    }

    let inputs = notation.inputs();
    let output = notation.output();

    let sub_a = &inputs[0];
    let sub_b = &inputs[1];

    // Must have exactly 2 indices each
    if sub_a.explicit_count() != 2 || sub_b.explicit_count() != 2 {
        return None;
    }

    // Output must have exactly 2 indices
    if output.explicit_count() != 2 {
        return None;
    }

    // Collect indices
    let indices_a: Vec<char> = sub_a.named_indices().collect();
    let indices_b: Vec<char> = sub_b.named_indices().collect();
    let indices_out: Vec<char> = output.named_indices().collect();

    let set_a: BTreeSet<char> = indices_a.iter().copied().collect();
    let set_b: BTreeSet<char> = indices_b.iter().copied().collect();
    let set_out: BTreeSet<char> = indices_out.iter().copied().collect();

    // Find contracted index (in both inputs, not in output)
    let common: BTreeSet<char> = set_a.intersection(&set_b).copied().collect();
    let contracted: Vec<char> = common.difference(&set_out).copied().collect();

    if contracted.len() != 1 {
        return None;
    }

    let k_index = contracted[0];

    // Find M index (in A and output, not in B)
    let m_candidates: Vec<char> = set_a
        .difference(&set_b)
        .filter(|c| set_out.contains(c))
        .copied()
        .collect();

    if m_candidates.len() != 1 {
        return None;
    }
    let m_index = m_candidates[0];

    // Find N index (in B and output, not in A)
    let n_candidates: Vec<char> = set_b
        .difference(&set_a)
        .filter(|c| set_out.contains(c))
        .copied()
        .collect();

    if n_candidates.len() != 1 {
        return None;
    }
    let n_index = n_candidates[0];

    // Determine transposition
    // Standard form: A[M, K], B[K, N] -> C[M, N]
    // A is transposed if K comes before M
    // B is transposed if N comes before K

    let m_pos_a = indices_a.iter().position(|&c| c == m_index).unwrap();
    let k_pos_a = indices_a.iter().position(|&c| c == k_index).unwrap();
    let k_pos_b = indices_b.iter().position(|&c| c == k_index).unwrap();
    let n_pos_b = indices_b.iter().position(|&c| c == n_index).unwrap();

    let transpose_a = k_pos_a < m_pos_a;  // K before M means A^T
    let transpose_b = n_pos_b < k_pos_b;  // N before K means B^T

    // Find positions in output
    let m_dim = indices_out.iter().position(|&c| c == m_index).unwrap();
    let n_dim = indices_out.iter().position(|&c| c == n_index).unwrap();

    Some(MatmulConfig {
        transpose_a,
        transpose_b,
        batch_dims: Vec::new(),
        m_dim,
        n_dim,
        k_index,
    })
}

/// Checks if the notation represents a batched matrix multiplication.
///
/// Batched matmul: `bij,bjk->bik` or with multiple batch dimensions.
pub fn is_batched_matmul(notation: &EinsumNotation) -> Option<MatmulConfig> {
    if !notation.is_binary() {
        return None;
    }

    let inputs = notation.inputs();
    let output = notation.output();

    let sub_a = &inputs[0];
    let sub_b = &inputs[1];

    // Must have at least 3 indices each (batch + matmul)
    if sub_a.explicit_count() < 3 || sub_b.explicit_count() < 3 {
        return None;
    }

    // Collect indices
    let indices_a: Vec<char> = sub_a.named_indices().collect();
    let indices_b: Vec<char> = sub_b.named_indices().collect();
    let indices_out: Vec<char> = output.named_indices().collect();

    let set_a: BTreeSet<char> = indices_a.iter().copied().collect();
    let set_b: BTreeSet<char> = indices_b.iter().copied().collect();
    let set_out: BTreeSet<char> = indices_out.iter().copied().collect();

    // Find batch indices (in all three: A, B, and output)
    let common_ab: BTreeSet<char> = set_a.intersection(&set_b).copied().collect();
    let batch_indices: Vec<char> = common_ab
        .intersection(&set_out)
        .copied()
        .collect();

    if batch_indices.is_empty() {
        return None;
    }

    // Find contracted index (in both A and B, not in output)
    let contracted: Vec<char> = common_ab
        .difference(&set_out)
        .copied()
        .collect();

    if contracted.len() != 1 {
        return None;
    }

    let k_index = contracted[0];

    // Find M index (in A and output, not in B, not batch)
    let batch_set: BTreeSet<char> = batch_indices.iter().copied().collect();
    let m_candidates: Vec<char> = set_a
        .difference(&set_b)
        .filter(|c| set_out.contains(c) && !batch_set.contains(c))
        .copied()
        .collect();

    if m_candidates.len() != 1 {
        return None;
    }
    let m_index = m_candidates[0];

    // Find N index (in B and output, not in A, not batch)
    let n_candidates: Vec<char> = set_b
        .difference(&set_a)
        .filter(|c| set_out.contains(c) && !batch_set.contains(c))
        .copied()
        .collect();

    if n_candidates.len() != 1 {
        return None;
    }
    let n_index = n_candidates[0];

    // Determine batch dimension positions in output
    let batch_dims: Vec<usize> = batch_indices
        .iter()
        .filter_map(|&c| indices_out.iter().position(|&x| x == c))
        .collect();

    // Verify batch dims are at the beginning (common pattern)
    // Actually, we should support any batch dim position, but for now require leading

    // Determine transposition (ignoring batch dims)
    // Find positions of M, K in A (after batch)
    let non_batch_a: Vec<char> = indices_a
        .iter()
        .filter(|c| !batch_set.contains(c))
        .copied()
        .collect();
    let non_batch_b: Vec<char> = indices_b
        .iter()
        .filter(|c| !batch_set.contains(c))
        .copied()
        .collect();

    if non_batch_a.len() != 2 || non_batch_b.len() != 2 {
        return None;
    }

    let m_pos_a = non_batch_a.iter().position(|&c| c == m_index).unwrap();
    let k_pos_a = non_batch_a.iter().position(|&c| c == k_index).unwrap();
    let k_pos_b = non_batch_b.iter().position(|&c| c == k_index).unwrap();
    let n_pos_b = non_batch_b.iter().position(|&c| c == n_index).unwrap();

    let transpose_a = k_pos_a < m_pos_a;
    let transpose_b = n_pos_b < k_pos_b;

    // Find positions in output (after batch)
    let non_batch_out: Vec<char> = indices_out
        .iter()
        .filter(|c| !batch_set.contains(c))
        .copied()
        .collect();

    if non_batch_out.len() != 2 {
        return None;
    }

    let m_dim = batch_dims.len() + non_batch_out.iter().position(|&c| c == m_index).unwrap();
    let n_dim = batch_dims.len() + non_batch_out.iter().position(|&c| c == n_index).unwrap();

    Some(MatmulConfig {
        transpose_a,
        transpose_b,
        batch_dims,
        m_dim,
        n_dim,
        k_index,
    })
}

/// Extracts matmul configuration (handles both batched and non-batched).
pub fn extract_matmul_config(notation: &EinsumNotation) -> Option<MatmulConfig> {
    is_batched_matmul(notation).or_else(|| is_matmul(notation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notation::parse_einsum;

    #[test]
    fn test_standard_matmul() {
        let notation = parse_einsum("ij,jk->ik").unwrap();
        let config = is_matmul(&notation).unwrap();

        assert!(!config.transpose_a);
        assert!(!config.transpose_b);
        assert!(config.batch_dims.is_empty());
        assert_eq!(config.k_index, 'j');
    }

    #[test]
    fn test_matmul_transpose_a() {
        // ji,jk->ik means A^T @ B
        let notation = parse_einsum("ji,jk->ik").unwrap();
        let config = is_matmul(&notation).unwrap();

        assert!(config.transpose_a);
        assert!(!config.transpose_b);
    }

    #[test]
    fn test_matmul_transpose_b() {
        // ij,kj->ik means A @ B^T
        let notation = parse_einsum("ij,kj->ik").unwrap();
        let config = is_matmul(&notation).unwrap();

        assert!(!config.transpose_a);
        assert!(config.transpose_b);
    }

    #[test]
    fn test_batched_matmul() {
        let notation = parse_einsum("bij,bjk->bik").unwrap();
        let config = is_batched_matmul(&notation).unwrap();

        assert!(!config.transpose_a);
        assert!(!config.transpose_b);
        assert_eq!(config.batch_dims, vec![0]);
    }

    #[test]
    fn test_multi_batch_matmul() {
        let notation = parse_einsum("abij,abjk->abik").unwrap();
        let config = is_batched_matmul(&notation).unwrap();

        assert_eq!(config.batch_dims.len(), 2);
    }

    #[test]
    fn test_attention_scores() {
        // bhqd,bhkd->bhqk (Q @ K^T with batch dims b,h)
        let notation = parse_einsum("bhqd,bhkd->bhqk").unwrap();
        let config = is_batched_matmul(&notation).unwrap();

        assert!(!config.transpose_a);
        assert!(config.transpose_b);  // kd -> K^T
        assert_eq!(config.batch_dims.len(), 2);
        assert_eq!(config.k_index, 'd');
    }

    #[test]
    fn test_not_matmul() {
        // Three inputs - not matmul
        let notation = parse_einsum("ij,jk,kl->il").unwrap();
        assert!(is_matmul(&notation).is_none());
        assert!(is_batched_matmul(&notation).is_none());
    }
}
