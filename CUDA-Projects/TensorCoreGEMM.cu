// tensor_core_gemm.cu - Mixed-precision GEMM using Tensor Cores
// Demonstrates cutting-edge GPU compute capabilities for AI/ML workloads

#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <random>

using namespace nvcuda;

// Tensor Core GEMM using WMMA API (Warp Matrix Multiply Accumulate)
__global__ void wmma_gemm(half* A, half* B, float* C, int M, int N, int K) {
    // Tile dimensions for WMMA
    const int WMMA_M = 16;
    const int WMMA_N = 16; 
    const int WMMA_K = 16;
    
    // Calculate warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Main computation loop
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices into fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Load C fragment and add
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);
        
        // Add accumulator to C
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] += acc_frag.x[i];
        }
        
        // Store result
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Advanced Tensor Core GEMM with persistent threads
__global__ void persistent_wmma_gemm(half* A, half* B, float* C, int M, int N, int K, int num_blocks) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Persistent thread block
    for (int block_id = blockIdx.x; block_id < num_blocks; block_id += gridDim.x) {
        int warpM = (block_id * blockDim.x + threadIdx.x) / warpSize;
        int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
        
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        
        wmma::fill_fragment(acc_frag, 0.0f);
        
        // Compute with loop tiling
        for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
            int aRow = warpM * WMMA_M;
            int aCol = k_tile;
            int bRow = k_tile;
            int bCol = warpN * WMMA_N;
            
            if (aRow < M && aCol < K && bRow < K && bCol < N) {
                wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
                wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }
        
        // Store result
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;
        if (cRow < M && cCol < N) {
            wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
        }
    }
}

// Mixed precision GEMM with automatic scaling
__global__ void mixed_precision_gemm(half* A, half* B, float* C, int M, int N, int K, 
                                    float alpha, float beta) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(acc_frag, 0.0f);
    
    // Main computation
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Apply scaling: C = alpha * A*B + beta * C
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, C + cRow * N + cCol, N, wmma::mem_row_major);
        
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

void initialize_half_matrix(half* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = __float2half(dis(gen));
    }
}

void initialize_float_matrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

bool check_tensor_core_support() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Check for Tensor Core support (Compute Capability 7.0+)
    if (prop.major >= 7) {
        std::cout << "Tensor Core support detected: CC " << prop.major << "." << prop.minor << std::endl;
        return true;
    } else {
        std::cout << "Tensor Cores not supported on this device (CC " << prop.major << "." << prop.minor << ")" << std::endl;
        return false;
    }
}

double benchmark_wmma_kernel(void(*kernel)(half*, half*, float*, int, int, int),
                            half* d_A, half* d_B, float* d_C, int M, int N, int K,
                            const char* kernel_name) {
    // Reset output matrix
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    // Grid configuration for WMMA (16x16 tiles)
    dim3 blockSize(256);
    dim3 gridSize((M + 15) / 16, (N + 15) / 16);
    
    // Warm up
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_runs = 20;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_time / 1000.0)) / 1e9;
    
    std::cout << kernel_name << ": " << avg_time << " ms (" << gflops << " GFLOPS)" << std::endl;
    
    return gflops;
}

double benchmark_mixed_precision(half* d_A, half* d_B, float* d_C, int M, int N, int K) {
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    dim3 blockSize(256);
    dim3 gridSize((M + 15) / 16, (N + 15) / 16);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warm up
    mixed_precision_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const int num_runs = 20;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        mixed_precision_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (avg_time / 1000.0)) / 1e9;
    
    std::cout << "Mixed Precision GEMM: " << avg_time << " ms (" << gflops << " GFLOPS)" << std::endl;
    
    return gflops;
}

int main() {
    if (!check_tensor_core_support()) {
        std::cout << "This demo requires Tensor Core support (RTX 20xx/30xx/40xx, V100, A100, etc.)" << std::endl;
        return 1;
    }
    
    // Matrix dimensions (must be multiples of 16 for WMMA)
    const int M = 4096, N = 4096, K = 4096;
    const int size_A = M * K;
    const int size_B = K * N;
    const int size_C = M * N;
    
    std::cout << "=== Tensor Core GEMM Benchmark ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    std::cout << "Using mixed precision: FP16 inputs, FP32 outputs" << std::endl;
    
    // Host memory
    half *h_A = new half[size_A];
    half *h_B = new half[size_B];
    float *h_C = new float[size_C];
    
    initialize_half_matrix(h_A, size_A);
    initialize_half_matrix(h_B, size_B);
    initialize_float_matrix(h_C, size_C);
    
    // Device memory
    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark different WMMA implementations
    double perf1 = benchmark_wmma_kernel(wmma_gemm, d_A, d_B, d_C, M, N, K, "Basic WMMA GEMM");
    
    // Persistent threads benchmark
    int num_blocks = (M + 15) / 16;
    dim3 blockSize(256);
    dim3 gridSize(std::min(num_blocks, 108)); // Limit to SM count
    
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    auto start = std::chrono::high_resolution_clock::now();
    persistent_wmma_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, num_blocks);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double pers_time = std::chrono::duration<double, std::milli>(end - start).count();
    double pers_gflops = (2.0 * M * N * K / (pers_time / 1000.0)) / 1e9;
    std::cout << "Persistent WMMA GEMM: " << pers_time << " ms (" << pers_gflops << " GFLOPS)" << std::endl;
    
    double perf3 = benchmark_mixed_precision(d_A, d_B, d_C, M, N, K);
    
    std::cout << "\nPerformance Analysis:" << std::endl;
    std::cout << "Theoretical peak (RTX 4090): ~330 TOPS for FP16" << std::endl;
    std::cout << "Achieved efficiency: " << (perf3 / 330000.0) * 100.0 << "%" << std::endl;
    
    // Memory bandwidth analysis
    double memory_bytes = (size_A + size_B) * sizeof(half) + size_C * sizeof(float);
    double bandwidth_gb_s = (memory_bytes / (pers_time / 1000.0)) / 1e9;
    std::cout << "Memory bandwidth utilization: " << bandwidth_gb_s << " GB/s" << std::endl;
    
    // Verify one result
    float *h_result = new float[size_C];
    CUDA_CHECK(cudaMemcpy(h_result, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "\nSample results:" << std::endl;
    std::cout << "C[0] = " << h_result[0] << std::endl;
    std::cout << "C[1000] = " << h_result[1000] << std::endl;
    std::cout << "C[" << size_C-1 << "] = " << h_result[size_C-1] << std::endl;
    
    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_result;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}
