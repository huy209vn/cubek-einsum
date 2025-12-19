//! Einsum benchmark suite.
//!
//! Tests both f32 and f16 (tensor core) performance across diverse patterns:
//! - Standard matmul/batch matmul (fast path)
//! - Tensor contractions (3D+)
//! - Novel patterns (bilinear forms, metric tensors, etc.)
use cubecl::{
    benchmark::{Benchmark, BenchmarkDurations, TimingMethod},
    future,
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek_einsum::{einsum, EinsumConfig, ContractionStrategy};
use cubek::random::random_uniform;
use half::f16;

/// Einsum benchmark with generic element type.
#[allow(dead_code)]
struct EinsumBench<R: Runtime, E: CubePrimitive + Numeric> {
    notation: &'static str,
    shapes: Vec<Vec<usize>>,
    strategy: ContractionStrategy,
    device: R::Device,
    client: ComputeClient<R>,
    _phantom: std::marker::PhantomData<E>,
}

impl<R: Runtime, E: CubePrimitive + Numeric> Benchmark for EinsumBench<R, E> {
    type Input = (Vec<TensorHandle<R>>, TensorHandle<R>);
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let client = R::client(&self.device);
        let dtype = E::as_type_native_unchecked();

        // Create input tensors
        let inputs: Vec<TensorHandle<R>> = self.shapes.iter().map(|shape| {
            let tensor = TensorHandle::empty(&client, shape.clone(), dtype);
            // Use f32 for random generation, converted internally
            random_uniform(&client, 0.0f32, 1.0f32, tensor.as_ref(), dtype).unwrap();
            tensor
        }).collect();

        // Compute output shape based on notation
        let output_shape = compute_output_shape(&self.notation, &self.shapes);
        let output = TensorHandle::empty(&client, output_shape, dtype);

        (inputs, output)
    }

    fn execute(&self, (inputs, mut output): Self::Input) -> Result<Self::Output, String> {
        let input_refs: Vec<&TensorHandle<R>> = inputs.iter().collect();
        let config = EinsumConfig {
            strategy: self.strategy,
            use_tensor_cores: true,
            autotune: false,
            validate_shapes: false,
        };

        einsum::<R, E>(
            &self.client,
            self.notation,
            &input_refs,
            &mut output,
            Some(config),
        ).map_err(|e| format!("{:?}", e))
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);
        let shape_str: Vec<String> = self.shapes.iter()
            .map(|s| format!("{:?}", s))
            .collect();
        let type_name = std::any::type_name::<E>().split("::").last().unwrap_or("?");
        format!(
            "{}-einsum-{}-{}-shapes[{}]-{:?}",
            R::name(&client),
            type_name,
            self.notation.replace(",", "_").replace("->", "_to_"),
            shape_str.join(","),
            self.strategy
        ).to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync()).unwrap()
    }

    fn profile(&self, args: Self::Input) -> Result<cubecl::benchmark::ProfileDuration, String> {
        self.client
            .profile(|| self.execute(args), "einsum-bench")
            .map(|it| it.1)
            .map_err(|err| format!("{err:?}"))
    }
}

/// Computes output shape from einsum notation and input shapes.
fn compute_output_shape(notation: &str, shapes: &[Vec<usize>]) -> Vec<usize> {
    use std::collections::HashMap;

    // Parse notation
    let parts: Vec<&str> = notation.split("->").collect();
    if parts.len() != 2 {
        // If no '->' provided, fall back to scalar (our benchmarks always use explicit ->).
        return vec![1];
    }

    let inputs_str = parts[0];
    let output_str = parts[1];

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    let input_subscripts: Vec<&str> = inputs_str.split(',').collect();

    for (i, subscript) in input_subscripts.iter().enumerate() {
        if i >= shapes.len() {
            break;
        }
        for (j, c) in subscript.chars().enumerate() {
            if j < shapes[i].len() && c.is_alphabetic() {
                dim_map.insert(c, shapes[i][j]);
            }
        }
    }

    // Build output shape
    if output_str.is_empty() {
        vec![1] // Scalar
    } else {
        output_str.chars()
            .filter(|c| c.is_alphabetic())
            .filter_map(|c| dim_map.get(&c).copied())
            .collect()
    }
}

/// Estimates FLOPs for an einsum operation.
///
/// For binary operations (2 inputs), uses the standard matmul formula: 2*M*K*N.
/// For chain contractions (3+ inputs), estimates pairwise contraction costs
/// since actual execution is done pairwise, not as a single giant contraction.
fn estimate_flops(notation: &str, shapes: &[Vec<usize>]) -> u64 {
    use std::collections::HashMap;

    let parts: Vec<&str> = notation.split("->").collect();
    let inputs_str = parts[0];
    let output_str = if parts.len() == 2 { parts[1] } else { "" };

    let input_subscripts: Vec<&str> = inputs_str.split(',').collect();

    // Build dimension map
    let mut dim_map: HashMap<char, usize> = HashMap::new();
    for (i, subscript) in input_subscripts.iter().enumerate() {
        if i >= shapes.len() {
            break;
        }
        for (j, c) in subscript.chars().enumerate() {
            if j < shapes[i].len() && c.is_alphabetic() {
                dim_map.insert(c, shapes[i][j]);
            }
        }
    }

    // For 2 inputs (standard case), use precise formula
    if input_subscripts.len() == 2 {
        let output_indices: Vec<char> = output_str.chars().filter(|c| c.is_alphabetic()).collect();

        // Find contracted indices
        let set_a: std::collections::HashSet<char> = input_subscripts[0].chars().filter(|c| c.is_alphabetic()).collect();
        let set_b: std::collections::HashSet<char> = input_subscripts[1].chars().filter(|c| c.is_alphabetic()).collect();
        let output_set: std::collections::HashSet<char> = output_indices.iter().copied().collect();

        let contracted: Vec<char> = set_a.intersection(&set_b)
            .filter(|c| !output_set.contains(c))
            .copied()
            .collect();

        // FLOPs = 2 * M * K * N where K is product of contracted dims
        let m: u64 = set_a.difference(&set_b)
            .filter(|c| output_set.contains(c))
            .map(|c| dim_map.get(c).copied().unwrap_or(1) as u64)
            .product::<u64>();
        let m = if m == 0 { 1 } else { m };

        let n: u64 = set_b.difference(&set_a)
            .filter(|c| output_set.contains(c))
            .map(|c| dim_map.get(c).copied().unwrap_or(1) as u64)
            .product::<u64>();
        let n = if n == 0 { 1 } else { n };

        let k: u64 = contracted.iter()
            .map(|c| dim_map.get(c).copied().unwrap_or(1) as u64)
            .product::<u64>();
        let k = if k == 0 { 1 } else { k };

        // Include batch dimensions
        let batch: u64 = output_indices.iter()
            .filter(|c| set_a.contains(c) && set_b.contains(c))
            .map(|c| dim_map.get(c).copied().unwrap_or(1) as u64)
            .product::<u64>();
        let batch = if batch == 0 { 1 } else { batch };

        return 2 * batch * m * k * n;
    }

    // For 3+ inputs (chain contractions), estimate pairwise costs
    // This is approximate but much more accurate than the naive formula
    if input_subscripts.len() >= 3 {
        let mut total_flops: u64 = 0;

        // Estimate each pairwise contraction
        // For ij,jk,kl->il: (ij,jk)->ik costs 2*i*j*k, then (ik,kl)->il costs 2*i*k*l
        for i in 0..input_subscripts.len() - 1 {
            let sub_a = input_subscripts[i];
            let sub_b = input_subscripts[i + 1];

            let set_a: std::collections::HashSet<char> = sub_a.chars().filter(|c| c.is_alphabetic()).collect();
            let set_b: std::collections::HashSet<char> = sub_b.chars().filter(|c| c.is_alphabetic()).collect();

            // Contracted index between adjacent tensors
            let contracted: Vec<char> = set_a.intersection(&set_b).copied().collect();

            let m: u64 = set_a.difference(&set_b)
                .map(|c| dim_map.get(c).copied().unwrap_or(1) as u64)
                .product::<u64>();
            let m = if m == 0 { 1 } else { m };

            let n: u64 = set_b.difference(&set_a)
                .map(|c| dim_map.get(c).copied().unwrap_or(1) as u64)
                .product::<u64>();
            let n = if n == 0 { 1 } else { n };

            let k: u64 = contracted.iter()
                .map(|c| dim_map.get(c).copied().unwrap_or(1) as u64)
                .product::<u64>();
            let k = if k == 0 { 1 } else { k };

            total_flops += 2 * m * k * n;
        }

        return total_flops;
    }

    // Single input (reduction/transpose) - minimal compute
    shapes.get(0)
        .map(|s| s.iter().map(|&d| d as u64).product::<u64>())
        .unwrap_or(0)
}

/// Runs a single einsum benchmark and reports both kernel TFLOPS and system timing.
fn run_one<R: Runtime, E: CubePrimitive + Numeric + 'static>(
    device: R::Device,
    notation: &'static str,
    shapes: Vec<Vec<usize>>,
    strategy: ContractionStrategy,
) -> Result<(BenchmarkDurations, f64), String> {
    let client = R::client(&device);

    let bench = EinsumBench::<R, E> {
        notation,
        shapes: shapes.clone(),
        strategy,
        client: client.clone(),
        device: device.clone(),
        _phantom: std::marker::PhantomData,
    };

    let type_name = std::any::type_name::<E>().split("::").last().unwrap_or("?");
    println!("Einsum: {} with shapes {:?} [{}]", notation, shapes, type_name);
    println!("{}", bench.name());

    // -------------------------
    // Warmup
    // -------------------------
    let warm_args = bench.prepare();
    let _ = bench.execute(warm_args);
    bench.sync();

    // -------------------------
    // Kernel profiling
    // -------------------------
    let profile_args = bench.prepare();
    let profile_duration = bench.profile(profile_args)
        .map_err(|e| format!("profiling failed: {:?}", e))?;

    let ticks = future::block_on(profile_duration.resolve());
    let kernel_secs = ticks.duration().as_secs_f64();

    // -------------------------
    // System timing
    // -------------------------
    match bench.run(TimingMethod::System) {
        Ok(val) => {
            let flops = estimate_flops(notation, &shapes);
            let secs = kernel_secs.max(1e-12);
            let tflops = flops as f64 / (secs * 1e12);

            println!("TFLOPS (kernel): {:.3}", tflops);
            println!("Times (system): {val}");

            Ok((val, tflops))
        }
        Err(err) => Err(format!("{err:?}")),
    }
}
#[allow(unused)]
fn bench_matmul<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Matrix Multiplication (Fast Path) ===");
    // Use larger sizes for better GPU utilization
    for (m, k, n) in [(2048, 2048, 2048), (4096, 4096, 4096)] {
        let _ = run_one::<R, E>(
            device.clone(),
            "ij,jk->ik",
            vec![vec![m, k], vec![k, n]],
            ContractionStrategy::Auto,
        );
    }
}

#[allow(unused)]
fn bench_batched_matmul<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Batched Matrix Multiplication (Fast Path) ===");
    // Larger batches with reasonable matrix sizes
    for (b, m, k, n) in [(16, 512, 512, 512), (64, 256, 256, 256)] {
        let _ = run_one::<R, E>(
            device.clone(),
            "bij,bjk->bik",
            vec![vec![b, m, k], vec![b, k, n]],
            ContractionStrategy::Auto,
        );
    }
}

#[allow(unused)]
fn bench_chain_contraction<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Chain Contraction (Multi-Step) ===");

    // 3-tensor chain: A @ B @ C
    let _ = run_one::<R, E>(
        device.clone(),
        "ij,jk,kl->il",
        vec![vec![512, 1024], vec![1024, 512], vec![512, 256]],
        ContractionStrategy::Auto,
    );

    // Compare greedy vs optimal with skewed sizes
    println!("\n--- Greedy vs Optimal Path ---");
    let _ = run_one::<R, E>(
        device.clone(),
        "ij,jk,kl->il",
        vec![vec![32, 512], vec![512, 2048], vec![2048, 32]],
        ContractionStrategy::Greedy,
    );
    let _ = run_one::<R, E>(
        device.clone(),
        "ij,jk,kl->il",
        vec![vec![32, 512], vec![512, 2048], vec![2048, 32]],
        ContractionStrategy::Optimal,
    );
}

#[allow(unused)]
fn bench_reductions<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Reductions (Fast Path) ===");

    // Sum all elements
    let _ = run_one::<R, E>(
        device.clone(),
        "ij->",
        vec![vec![2048, 2048]],
        ContractionStrategy::Auto,
    );

    // Sum along axis
    let _ = run_one::<R, E>(
        device.clone(),
        "ij->i",
        vec![vec![2048, 2048]],
        ContractionStrategy::Auto,
    );

    // Trace
    let _ = run_one::<R, E>(
        device.clone(),
        "ii->",
        vec![vec![2048, 2048]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_elementwise<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Element-wise Operations ===");

    // Hadamard product
    let _ = run_one::<R, E>(
        device.clone(),
        "ij,ij->ij",
        vec![vec![2048, 2048], vec![2048, 2048]],
        ContractionStrategy::Auto,
    );

    // Outer product
    let _ = run_one::<R, E>(
        device.clone(),
        "i,j->ij",
        vec![vec![2048], vec![2048]],
        ContractionStrategy::Auto,
    );

    // Dot product
    let _ = run_one::<R, E>(
        device.clone(),
        "i,i->",
        vec![vec![4 * 1024 * 1024], vec![4 * 1024 * 1024]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_attention_pattern<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Attention Patterns ===");

    // Attention scores: Q @ K^T
    let _ = run_one::<R, E>(
        device.clone(),
        "bhqd,bhkd->bhqk",
        vec![vec![16, 12, 512, 64], vec![16, 12, 512, 64]],
        ContractionStrategy::Auto,
    );

    // Attention output: scores @ V
    let _ = run_one::<R, E>(
        device.clone(),
        "bhqk,bhkd->bhqd",
        vec![vec![16, 12, 512, 512], vec![16, 12, 512, 64]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_large_attention<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Large Attention Patterns (Transformer Scale) ===");

    // Realistic GPT-2 scale: batch=16, heads=16, seq=1024, dim=64
    let _ = run_one::<R, E>(
        device.clone(),
        "bhqd,bhkd->bhqk",
        vec![vec![16, 16, 1024, 64], vec![16, 16, 1024, 64]],
        ContractionStrategy::Auto,
    );

    // Wide attention: batch=8, heads=32, seq=512, dim=128
    let _ = run_one::<R, E>(
        device.clone(),
        "bhqd,bhkd->bhqk",
        vec![vec![8, 32, 512, 128], vec![8, 32, 512, 128]],
        ContractionStrategy::Auto,
    );
}

// ============================================================================
// NOVEL EINSUM PATTERNS - True power of Einstein notation
// ============================================================================

#[allow(unused)]
fn bench_tensor_network<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Tensor Network Contractions ===");

    // Batched tensor contraction: batch index shared
    let _ = run_one::<R, E>(
        device.clone(),
        "bijk,bkjl->bil",
        vec![vec![16, 64, 128, 64], vec![16, 64, 128, 64]],
        ContractionStrategy::Auto,
    );

    // Multi-head style tensor contraction
    let _ = run_one::<R, E>(
        device.clone(),
        "bhij,bhjk->bhik",
        vec![vec![32, 8, 256, 128], vec![32, 8, 128, 256]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_bilinear_forms<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Bilinear Forms & Gram Matrices ===");

    // Gram matrix computation: X X^T (kernel matrix, covariance)
    let _ = run_one::<R, E>(
        device.clone(),
        "ik,jk->ij",
        vec![vec![1024, 512], vec![1024, 512]],
        ContractionStrategy::Auto,
    );

    // Batched Gram: feature covariance per batch
    let _ = run_one::<R, E>(
        device.clone(),
        "bik,bjk->bij",
        vec![vec![32, 256, 128], vec![32, 256, 128]],
        ContractionStrategy::Auto,
    );

    // Inner product matrix: X^T Y (cross-covariance)
    let _ = run_one::<R, E>(
        device.clone(),
        "ki,kj->ij",
        vec![vec![512, 1024], vec![512, 1024]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_physics_patterns<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Physics-Inspired Patterns ===");

    // Tensor double contraction (like : operator) - smaller to avoid dispatch limits
    let _ = run_one::<R, E>(
        device.clone(),
        "abcd,abcd->",
        vec![vec![32, 32, 32, 32], vec![32, 32, 32, 32]],
        ContractionStrategy::Auto,
    );

    // Batched tensor hadamard + sum - smaller to avoid dispatch limits
    let _ = run_one::<R, E>(
        device.clone(),
        "bijkl,bijkl->b",
        vec![vec![16, 16, 16, 16, 16], vec![16, 16, 16, 16, 16]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_ml_patterns<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Machine Learning Patterns ===");

    // Graph neural network: adjacency @ features
    // A[n,n] @ X[n,d] -> out[n,d]
    let _ = run_one::<R, E>(
        device.clone(),
        "nm,md->nd",
        vec![vec![4096, 4096], vec![4096, 64]],
        ContractionStrategy::Auto,
    );

    // Einsum attention variant (Q @ K^T batched)
    let _ = run_one::<R, E>(
        device.clone(),
        "bhsd,bhrd->bhsr",
        vec![vec![32, 8, 256, 64], vec![32, 8, 256, 64]],
        ContractionStrategy::Auto,
    );

    // Feature cross-correlation (channel mixing)
    let _ = run_one::<R, E>(
        device.clone(),
        "bchw,bdhw->bcd",
        vec![vec![16, 64, 64, 64], vec![16, 128, 64, 64]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_permutation_heavy<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Permutation & Diagonal Patterns ===");

    // Transpose (2D)
    let _ = run_one::<R, E>(
        device.clone(),
        "ij->ji",
        vec![vec![2048, 2048]],
        ContractionStrategy::Auto,
    );

    // Trace: diagonal sum
    let _ = run_one::<R, E>(
        device.clone(),
        "ii->",
        vec![vec![2048, 2048]],
        ContractionStrategy::Auto,
    );

    // Batch diagonal extraction
    let _ = run_one::<R, E>(
        device.clone(),
        "bii->bi",
        vec![vec![64, 512, 512]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_broadcasting<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Broadcasting Patterns ===");

    // Broadcast vector to matrix: ij,j->ij
    let _ = run_one::<R, E>(
        device.clone(),
        "ij,j->ij",
        vec![vec![2048, 2048], vec![2048]],
        ContractionStrategy::Auto,
    );

    // Broadcast scalar to batch: b,->b (element-wise with scalar)
    let _ = run_one::<R, E>(
        device.clone(),
        "bi,i->bi",
        vec![vec![256, 2048], vec![2048]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn bench_complex_networks<R: Runtime, E: CubePrimitive + Numeric + 'static>(device: R::Device) {
    println!("\n=== Complex Contractions ===");

    // Batched contraction with two contracted indices
    let _ = run_one::<R, E>(
        device.clone(),
        "bijkl,bjklm->bim",
        vec![vec![8, 32, 64, 64, 32], vec![8, 64, 64, 32, 32]],
        ContractionStrategy::Auto,
    );

    // Outer product (no contraction)
    let _ = run_one::<R, E>(
        device.clone(),
        "i,j->ij",
        vec![vec![2048], vec![2048]],
        ContractionStrategy::Auto,
    );

    // Large matmul with unusual layout
    let _ = run_one::<R, E>(
        device.clone(),
        "ik,kj->ij",
        vec![vec![2048, 1024], vec![1024, 4096]],
        ContractionStrategy::Auto,
    );
}

#[allow(unused)]
fn run_all_benches<R: Runtime>(enable_f16: bool) {
    let device = R::Device::default();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              CubeK Einsum Benchmarks                         ║");
    println!("║    Showcasing novel tensor contraction patterns              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // f32 BENCHMARKS
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    f32 Benchmarks                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Core matmul patterns
    bench_matmul::<R, f32>(device.clone());
    bench_batched_matmul::<R, f32>(device.clone());
    bench_attention_pattern::<R, f32>(device.clone());
    bench_large_attention::<R, f32>(device.clone());

    // Novel patterns
    bench_tensor_network::<R, f32>(device.clone());
    bench_bilinear_forms::<R, f32>(device.clone());
    bench_ml_patterns::<R, f32>(device.clone());

    // =========================================================================
    // f16 BENCHMARKS (Tensor Cores on supported hardware)
    // =========================================================================
    if enable_f16 {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║              f16 Benchmarks (Tensor Cores)                   ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        // Core matmul with f16 - should see ~4x speedup with tensor cores
        bench_matmul::<R, f16>(device.clone());
        bench_batched_matmul::<R, f16>(device.clone());
        bench_attention_pattern::<R, f16>(device.clone());
        bench_large_attention::<R, f16>(device.clone());

        // Novel patterns with f16
        bench_tensor_network::<R, f16>(device.clone());
        bench_bilinear_forms::<R, f16>(device.clone());
        bench_ml_patterns::<R, f16>(device.clone());
    } else {
        println!("(Skipped f16 benchmarks)");
    }

    // =========================================================================
    // ADDITIONAL PATTERNS (f32 only for now)
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                  Additional Patterns (f32)                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    bench_chain_contraction::<R, f32>(device.clone());
    bench_reductions::<R, f32>(device.clone());
    bench_elementwise::<R, f32>(device.clone());
    bench_permutation_heavy::<R, f32>(device.clone());
    bench_broadcasting::<R, f32>(device.clone());
    bench_complex_networks::<R, f32>(device.clone());
    bench_physics_patterns::<R, f32>(device.clone());
}

fn main() {
    #[cfg(feature = "cuda")]
    run_all_benches::<cubecl::cuda::CudaRuntime>(true);

    #[cfg(feature = "wgpu")]
    run_all_benches::<cubecl::wgpu::WgpuRuntime>(false);
}
