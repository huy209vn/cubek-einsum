CubeK Einsum

SOTA Einstein Summation (Einsum) implementation for GPU tensor operations.
Overview

CubeK Einsum provides a high-level API for expressing complex tensor operations using Einstein summation notation, with automatic optimization and execution on GPU hardware.
Features

    Full einsum notation parsing - Standard NumPy-style syntax including ellipsis (...)
    Pattern recognition - Fast paths for common operations:
        Matrix multiplication (ij,jk->ik)
        Batched matrix multiplication (bij,bjk->bik)
        Transpose (ij->ji)
        Reduction (ij->i)
        Hadamard product (ij,ij->ij)
        Outer product (i,j->ij)
        Dot product (i,i->)
        Trace (ii->)
    Contraction path optimization - Greedy and dynamic programming algorithms
    Workspace management - Automatic allocation for multi-step contractions
    General contraction execution - Support for chain operations like ij,jk,kl->il

Usage
Basic Example

use cubek_einsum::einsum;

// Matrix multiplication
einsum::<R, f32>(&client, "ij,jk->ik", &[&a, &b], &mut c, None)?;

// Batched attention scores
einsum::<R, f32>(&client, "bhqd,bhkd->bhqk", &[&q, &k], &mut scores, None)?;

// Gram matrix (A @ B^T)
einsum::<R, f32>(&client, "ik,jk->ij", &[&a, &b], &mut gram, None)?;

// Reduction
einsum::<R, f32>(&client, "ij->i", &[&x], &mut row_sums, None)?;

Chain Contraction

// Chain of three matrix multiplications: A @ B @ C
einsum::<R, f32>(&client, "ij,jk,kl->il", &[&a, &b, &c], &mut result, None)?;

Notation Reference
Notation 	Operation 	Example
ij,jk->ik 	Matrix multiply 	C[i,k] = sum_j A[i,j] * B[j,k]
bij,bjk->bik 	Batched matmul 	Batch dimension preserved
ij->ji 	Transpose 	Dimension reorder
ii-> 	Trace 	Diagonal sum
ij,ij->ij 	Hadamard product 	Element-wise multiply
i,j->ij 	Outer product 	Rank expansion
ij->i 	Row sum 	Reduction over j
ij-> 	Total sum 	Full reduction
Performance

Benchmarked on RTX 3090:
Operation 	f16 	f32
Matmul 4096³ (ij,jk->ik) 	~84 TFLOPS 	~22 TFLOPS
Batched matmul (bij,bjk->bik) 	~80 TFLOPS 	~20 TFLOPS
Gram matrix (ik,jk->ij) 	~80 TFLOPS 	~20 TFLOPS

Performance matches the underlying cubek-matmul for recognized patterns.
Design

    Fast paths: Direct dispatch to optimized cubek kernels (matmul, reduce)
    Chain contractions: Automatic workspace management for intermediate results
    Automatic optimization: Intelligent path selection based on problem size
    Zero-copy transpose: Stride manipulation for permutations when possible

Testing

Run the test suite:

cd cubek/crates/cubek-einsum
cargo test --lib

Status

Initial implementation complete. Core functionality working with good performance for common patterns.
Known Limitations

    Tall-skinny matrix performance (e.g., 4096x4096 @ 4096x64) limited by underlying matmul kernel
    Complex tensor networks with poor arithmetic intensity will have low throughput

Future Work

    Branch and bound optimization for medium-sized expressions
    Autotuning framework with shape-aware caching
    Kernel fusion for adjacent operations
