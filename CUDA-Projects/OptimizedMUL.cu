// optimized_gemm.cu - Tiled GEMM with shared memory optimization
// Demonstrates advanced memory hierarchy usage and coalesced access patterns

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

#define TILE_SIZE 32
#define BLOCK_SIZE 32

// Optimized tiled GEMM kernel using shared memory
__global__ void tiled_gemm(float* A, float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Global indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory with bounds checking
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Double-buffered tiled GEMM for even better performance
__global__ void double_buffered_gemm(float* A, float* B, float* C, int M, int N, int K) {
    // Double buffering with shared memory
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    int buffer = 0;
    
    // Preload first tile
    if (row < M && tx < K) {
        As[buffer][ty][tx] = A[row * K + tx];
    } else {
        As[buffer][ty][tx] = 0.0f;
    }
    
    if (col < N && ty < K) {
        Bs[buffer][ty][tx] = B[ty * N + col];
    } else {
        Bs[buffer][ty][tx] = 0.0f;
    }
    
    __syncthreads();
    
    for (int t = 1; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int next_buffer = 1 - buffer;
        
        // Load next tile while computing current
        if (row < M && t * TILE_SIZE + tx < K) {
            As[next_buffer][ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[next_buffer][ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[next_buffer][ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[next_buffer][ty][tx] = 0.0f;
        }
        
        // Compute using current buffer
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[buffer][ty][k] * Bs[buffer][k][tx];
        }
        
        __syncthreads();
        buffer = next_buffer;
    }
    
    // Compute last tile
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[buffer][ty][k] * Bs[buffer][k][tx];
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Vectorized GEMM using float4 for memory coalescing
__global__ void vectorized_gemm(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Coalesced loading with vectorization where possible
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Unrolled inner loop for better instruction-level parallelism
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void initialize_matrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
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

double benchmark_kernel(void(*kernel)(float*, float*, float*, int, int, int),
                       float* d_A, float* d_B, float* d_C, int M, int N, int K,
                       const char* kernel_name) {
    // Reset output matrix
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
                  (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Warm up
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_runs = 10;
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

int main() {
    // Matrix dimensions - use larger sizes to show optimization benefits
    const int M = 2048, N = 2048, K = 2048;
    const int size_A = M * K;
    const int size_B = K * N;
    const int size_C = M * N;
    
    // Host memory
    float *h_A = new float[size_A];
    float *h_B = new float[size_B];
    float *h_C = new float[size_C];
    
    initialize_matrix(h_A, size_A);
    initialize_matrix(h_B, size_B);
    
    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "=== Optimized GEMM Benchmark ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    
    // Benchmark different implementations
    double perf1 = benchmark_kernel(tiled_gemm, d_A, d_B, d_C, M, N, K, "Tiled GEMM");
    double perf2 = benchmark_kernel(double_buffered_gemm, d_A, d_B, d_C, M, N, K, "Double-buffered GEMM");
    double perf3 = benchmark_kernel(vectorized_gemm, d_A, d_B, d_C, M, N, K, "Vectorized GEMM");
    
    std::cout << "\nPerformance improvements:" << std::endl;
    std::cout << "Double-buffered vs Tiled: " << perf2/perf1 << "x" << std::endl;
    std::cout << "Vectorized vs Tiled: " << perf3/perf1 << "x" << std::endl;
    
    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}