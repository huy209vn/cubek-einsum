//! Integration tests for cubek-einsum..
//!
//! These tests actually execute einsum operations on GPU.

use cubecl::prelude::*;
use cubecl::runtime::CudaRuntime;
use cubecl::client::ComputeClient;

use cubek_einsum::{einsum, EinsumConfig};
use cubek_einsum::notation::parse_einsum;

#[test]
fn test_matmul_integration() {
    let runtime = CudaRuntime::new().unwrap();
    let client = ComputeClient::new(&runtime).unwrap();

    // Create test tensors
    let a_shape = [10, 20];
    let b_shape = [20, 30];

    let mut a = TensorHandle::zeros(&client, a_shape, StorageType::F32);
    let mut b = TensorHandle::zeros(&client, b_shape, StorageType::F32);

    // Fill with some values (1.0 for simplicity)
    cubecl::std::tensor::fill_with(&mut a, |_| 1.0f32);
    cubecl::std::tensor::fill_with(&mut b, |_| 2.0f32);

    let mut c = TensorHandle::zeros(&client, [10, 30], StorageType::F32);

    // Execute einsum
    let result = einsum::<CudaRuntime, f32>(
        &client,
        "ij,jk->ik",
        &[&a, &b],
        &mut c,
        None,
    );

    assert!(result.is_ok());
}

#[test]
fn test_chain_contraction_integration() {
    let runtime = CudaRuntime::new().unwrap();
    let client = ComputeClient::new(&runtime).unwrap();

    // Create test tensors for chain: ij,jk,kl->il
    let a_shape = [10, 20];
    let b_shape = [20, 30];
    let c_shape = [30, 40];

    let mut a = TensorHandle::zeros(&client, a_shape, StorageType::F32);
    let mut b = TensorHandle::zeros(&client, b_shape, StorageType::F32);
    let mut c = TensorHandle::zeros(&client, c_shape, StorageType::F32);

    // Fill with some values
    cubecl::std::tensor::fill_with(&mut a, |_| 1.0f32);
    cubecl::std::tensor::fill_with(&mut b, |_| 2.0f32);
    cubecl::std::tensor::fill_with(&mut c, |_| 3.0f32);

    let mut result = TensorHandle::zeros(&client, [10, 40], StorageType::F32);

    // Execute chain contraction
    let result = einsum::<CudaRuntime, f32>(
        &client,
        "ij,jk,kl->il",
        &[&a, &b, &c],
        &mut result,
        None,
    );

    assert!(result.is_ok());
}

#[test]
fn test_batched_matmul_integration() {
    let runtime = CudaRuntime::new().unwrap();
    let client = ComputeClient::new(&runtime).unwrap();

    // Create batched tensors: bij,bjk->bik
    let a_shape = [5, 10, 20]; // batch=5, m=10, k=20
    let b_shape = [5, 20, 30]; // batch=5, k=20, n=30

    let mut a = TensorHandle::zeros(&client, a_shape, StorageType::F32);
    let mut b = TensorHandle::zeros(&client, b_shape, StorageType::F32);

    // Fill with some values
    cubecl::std::tensor::fill_with(&mut a, |_| 1.0f32);
    cubecl::std::tensor::fill_with(&mut b, |_| 2.0f32);

    let mut c = TensorHandle::zeros(&client, [5, 10, 30], StorageType::F32);

    // Execute batched einsum
    let result = einsum::<CudaRuntime, f32>(
        &client,
        "bij,bjk->bik",
        &[&a, &b],
        &mut c,
        None,
    );

    assert!(result.is_ok());
}

#[test]
fn test_reduction_integration() {
    let runtime = CudaRuntime::new().unwrap();
    let client = ComputeClient::new(&runtime).unwrap();

    // Create tensor for reduction: ij->i
    let a_shape = [10, 20];

    let mut a = TensorHandle::zeros(&client, a_shape, StorageType::F32);

    // Fill with some values
    cubecl::std::tensor::fill_with(&mut a, |_| 1.0f32);

    let mut result = TensorHandle::zeros(&client, [10], StorageType::F32);

    // Execute reduction
    let result = einsum::<CudaRuntime, f32>(
        &client,
        "ij->i",
        &[&a],
        &mut result,
        None,
    );

    assert!(result.is_ok());
}

#[test]
fn test_transpose_integration() {
    let runtime = CudaRuntime::new().unwrap();
    let client = ComputeClient::new(&runtime).unwrap();

    // Create tensor for transpose: ij->ji
    let a_shape = [10, 20];

    let mut a = TensorHandle::zeros(&client, a_shape, StorageType::F32);

    // Fill with some values
    cubecl::std::tensor::fill_with(&mut a, |_| 1.0f32);

    let mut result = TensorHandle::zeros(&client, [20, 10], StorageType::F32);

    // Execute transpose
    let result = einsum::<CudaRuntime, f32>(
        &client,
        "ij->ji",
        &[&a],
        &mut result,
        None,
    );

    assert!(result.is_ok());
}
"