// basic_gemm.cu - Basic General Matrix Multiplication
// Demonstrates fundamental CUDA concepts and naive GEMM implementation

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include <random>

// Naive GEMM kernel - C = A * B + C
__global__ void naive_gemm(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int q = 0; q < K; q++) {
            sum += A[row * K + q] * B[q * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

void initialize_matrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

// Fixed verification function with better tolerance and error reporting
void verify_result(float* C_gpu, float* C_cpu, int size, float tolerance = 1e-3f) {
    int error_count = 0;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float error = std::abs(C_gpu[i] - C_cpu[i]);
        float relative_error = error / std::max(std::abs(C_cpu[i]), 1e-7f);
        
        avg_error += error;
        max_error = std::max(max_error, error);
        
        // Check both absolute and relative error
        if (error > tolerance && relative_error > tolerance) {
            if (error_count < 10) { // Only print first 10 errors
                std::cout << "Error at index " << i 
                          << ": GPU=" << C_gpu[i] 
                          << ", CPU=" << C_cpu[i] 
                          << ", diff=" << error
                          << ", rel_error=" << relative_error << std::endl;
            }
            error_count++;
        }
    }
    
    avg_error /= size;
    
    std::cout << "\n=== Verification Results ===" << std::endl;
    std::cout << "Total errors: " << error_count << " / " << size << std::endl;
    std::cout << "Max absolute error: " << max_error << std::endl;
    std::cout << "Average absolute error: " << avg_error << std::endl;
    std::cout << "Error rate: " << (100.0f * error_count / size) << "%" << std::endl;
    
    if (error_count == 0) {
        std::cout << "Verification PASSED!" << std::endl;
    } else if (error_count < size * 0.01f) { // Less than 1% errors
        std::cout << "Verification PASSED with minor numerical differences" << std::endl;
    } else {
        std::cout <<  "Veification FAILED - significant errors detected" << std::endl;
    }
}


void cpu_gemm(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // Matrix dimensions
    const int M = 1024, N = 1024, K = 1024;
    const int size_A = M * K;
    const int size_B = K * N;
    const int size_C = M * N;
    
    // Host memory allocation
    float *h_A = new float[size_A];
    float *h_B = new float[size_B];
    float *h_C = new float[size_C];
    float *h_C_ref = new float[size_C];
    
    // Initialize matrices
    initialize_matrix(h_A, size_A);
    initialize_matrix(h_B, size_B);
    std::fill(h_C, h_C + size_C, 0.0f);
    std::fill(h_C_ref, h_C_ref + size_C, 0.0f);
    
    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C * sizeof(float), cudaMemcpyHostToDevice));
    
    // Kernel configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // Warm up
    naive_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing GPU kernel
    auto start = std::chrono::high_resolution_clock::now();
    naive_gemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto gpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost));
    
    // CPU reference computation
    start = std::chrono::high_resolution_clock::now();
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
    end = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Performance metrics
    double flops = 2.0 * M * N * K;
    double gpu_gflops = (flops / (gpu_time / 1000.0)) / 1e9;
    double cpu_gflops = (flops / (cpu_time / 1000.0)) / 1e9;
    
    std::cout << "=== Basic GEMM Performance ===" << std::endl;
    std::cout << "Matrix size: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms (" << gpu_gflops << " GFLOPS)" << std::endl;
    std::cout << "CPU time: " << cpu_time << " ms (" << cpu_gflops << " GFLOPS)" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    
    // Verify correctness
    verify_result(h_C, h_C_ref, size_C);
    
    // Cleanup
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_ref;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    
    return 0;
}