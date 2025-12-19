//! Fast path operation types.

use alloc::vec::Vec;

/// A recognized fast-path operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FastPath {
    /// Standard matrix multiplication.
    /// `ij,jk->ik` or variants with transposition.
    Matmul {
        /// Whether first input is transposed.
        transpose_a: bool,
        /// Whether second input is transposed.
        transpose_b: bool,
    },

    /// Batched matrix multiplication.
    /// `bij,bjk->bik` or with multiple batch dimensions.
    BatchedMatmul {
        /// Indices of batch dimensions (in input order).
        batch_dims: Vec<usize>,
        /// Whether first input is transposed (after batch dims).
        transpose_a: bool,
        /// Whether second input is transposed (after batch dims).
        transpose_b: bool,
    },

    /// Tensor reduction.
    /// `ij->i` (sum over j), `ijk->` (sum all), etc.
    Reduce {
        /// Axes to reduce over.
        axes: Vec<usize>,
        /// Type of reduction.
        op: ReduceOp,
    },

    /// Transpose (permutation of dimensions).
    /// `ij->ji`, `ijkl->jilk`, etc.
    Transpose {
        /// Permutation of dimensions.
        permutation: Vec<usize>,
    },

    /// Hadamard (element-wise) product.
    /// `ij,ij->ij`
    Hadamard,

    /// Outer product.
    /// `i,j->ij`
    OuterProduct,

    /// Dot product (inner product).
    /// `i,i->` or `ij,ij->`
    DotProduct,

    /// Trace of a matrix.
    /// `ii->`
    Trace,

    /// Diagonal extraction.
    /// `ii->i` or `bii->bi`
    DiagonalExtract,
}

/// Type of reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Mean,
}

impl FastPath {
    /// Returns true if this is a matmul variant.
    pub fn is_matmul(&self) -> bool {
        matches!(self, FastPath::Matmul { .. } | FastPath::BatchedMatmul { .. })
    }

    /// Returns true if this is a unary operation.
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            FastPath::Reduce { .. }
                | FastPath::Transpose { .. }
                | FastPath::Trace
                | FastPath::DiagonalExtract
        )
    }

    /// Returns true if this is a binary operation.
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            FastPath::Matmul { .. }
                | FastPath::BatchedMatmul { .. }
                | FastPath::Hadamard
                | FastPath::OuterProduct
                | FastPath::DotProduct
        )
    }

    /// Returns a human-readable name for this operation.
    pub fn name(&self) -> &'static str {
        match self {
            FastPath::Matmul { .. } => "matmul",
            FastPath::BatchedMatmul { .. } => "batched_matmul",
            FastPath::Reduce { .. } => "reduce",
            FastPath::Transpose { .. } => "transpose",
            FastPath::Hadamard => "hadamard",
            FastPath::OuterProduct => "outer_product",
            FastPath::DotProduct => "dot_product",
            FastPath::Trace => "trace",
            FastPath::DiagonalExtract => "diagonal_extract",
        }
    }
}
