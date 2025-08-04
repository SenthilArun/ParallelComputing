// convolution.cu - Optimized 2D convolution implementations
// Demonstrates image processing and CNN building blocks

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cmath>

// Naive 2D convolution kernel
__global__ void naive_conv2d(float* input, float* kernel, float* output,
                             int in_height, int in_width, int out_height, int out_width,
                             int kernel_size, int stride, int padding) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x < out_width && out_y < out_height) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = out_x * stride + kx - padding;
                int in_y = out_y * stride + ky - padding;
                
                if (in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height) {
                    sum += input[in_y * in_width + in_x] * kernel[ky * kernel_size + kx];
                }
            }
        }
        
        output[out_y * out_width + out_x] = sum;
    }
}

// Shared memory optimized convolution
#define TILE_SIZE 16
#define KERNEL_RADIUS 4  // Support up to 9x9 kernels

__global__ void shared_conv2d(float* input, float* kernel, float* output,
                              int in_height, int in_width, int out_height, int out_width,
                              int kernel_size, int stride, int padding) {
    __shared__ float shared_input[TILE_SIZE + 2*KERNEL_RADIUS][TILE_SIZE + 2*KERNEL_RADIUS];
    __shared__ float shared_kernel[9][9]; // Max 9x9 kernel
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + tx;
    int out_y = blockIdx.y * TILE_SIZE + ty;
    
    // Load kernel into shared memory (only first few threads)
    if (tx < kernel_size && ty < kernel_size) {
        shared_kernel[ty][tx] = kernel[ty * kernel_size + tx];
    }
    
    // Calculate input coordinates for shared memory loading
    int shared_size = TILE_SIZE + 2 * KERNEL_RADIUS;
    int loads_per_thread = (shared_size * shared_size + TILE_SIZE * TILE_SIZE - 1) / (TILE_SIZE * TILE_SIZE);
    
    for (int load = 0; load < loads_per_thread; load++) {
        int thread_id = ty * TILE_SIZE + tx;
        int total_thread_id = load * TILE_SIZE * TILE_SIZE + thread_id;
        
        if (total_thread_id < shared_size * shared_size) {
            int shared_y = total_thread_id / shared_size;
            int shared_x = total_thread_id % shared_size;
            
            int global_x = blockIdx.x * TILE_SIZE + shared_x - KERNEL_RADIUS;
            int global_y = blockIdx.y * TILE_SIZE + shared_y - KERNEL_RADIUS;
            
            if (global_x >= 0 && global_x < in_width && global_y >= 0 && global_y < in_height) {
                shared_input[shared_y][shared_x] = input[global_y * in_width + global_x];
            } else {
                shared_input[shared_y][shared_x] = 0.0f; // Zero padding
            }
        }
    }
    
    __syncthreads();
    
    // Compute convolution
    if (out_x < out_width && out_y < out_height) {
        float sum = 0.0f;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int shared_x = tx + kx;
                int shared_y = ty + ky;
                sum += shared_input[shared_y][shared_x] * shared_kernel[ky][kx];
            }
        }
        
        output[out_y * out_width + out_x] = sum;
    }
}

// Separable convolution for symmetric kernels (Gaussian blur, etc.)
__global__ void separable_conv_horizontal(float* input, float* temp, float* kernel_1d,
                                         int height, int width, int kernel_size, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int radius = kernel_size / 2;
        
        for (int k = 0; k < kernel_size; k++) {
            int input_x = x + k - radius - padding;
            if (input_x >= 0 && input_x < width) {
                sum += input[y * width + input_x] * kernel_1d[k];
            }
        }
        
        temp[y * width + x] = sum;
    }
}

__global__ void separable_conv_vertical(float* temp, float* output, float* kernel_1d,
                                       int height, int width, int kernel_size, int padding) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int radius = kernel_size / 2;
        
        for (int k = 0; k < kernel_size; k++) {
            int input_y = y + k - radius - padding;
            if (input_y >= 0 && input_y < height) {
                sum += temp[input_y * width + x] * kernel_1d[k];
            }
        }
        
        output[y * width + x] = sum;
    }
}

// Depthwise separable convolution (used in MobileNets)
__global__ void depthwise_conv2d(float* input, float* kernel, float* output,
                                int channels, int height, int width, int kernel_size) {
    int c = blockIdx.z;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (c < channels && x < width && y < height) {
        float sum = 0.0f;
        int radius = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int in_x = x + kx - radius;
                int in_y = y + ky - radius;
                
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    int input_idx = c * height * width + in_y * width + in_x;
                    int kernel_idx = c * kernel_size * kernel_size + ky * kernel_size + kx;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
        
        int output_idx = c * height * width + y * width + x;
        output[output_idx] = sum;
    }
}

// Winograd convolution (F(2x2, 3x3) - faster for 3x3 kernels)
__global__ void winograd_conv2d(float* input, float* kernel, float* output,
                               int height, int width, int out_height, int out_width) {
    // Winograd transformation matrices
    __shared__ float G[4][3]; // Kernel transform
    __shared__ float BT[4][4]; // Input transform
    __shared__ float AT[2][4]; // Output transform
    
    // Initialize transformation matrices (only first thread)
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // G matrix for F(2x2, 3x3)
        G[0][0] = 1.0f; G[0][1] = 0.0f; G[0][2] = 0.0f;
        G[1][0] = 0.5f; G[1][1] = 0.5f; G[1][2] = 0.5f;
        G[2][0] = 0.5f; G[2][1] = -0.5f; G[2][2] = 0.5f;
        G[3][0] = 0.0f; G[3][1] = 0.0f; G[3][2] = 1.0f;
        
        // BT matrix
        BT[0][0] = 1.0f; BT[0][1] = 0.0f; BT[0][2] = -1.0f; BT[0][3] = 0.0f;
        BT[1][0] = 0.0f; BT[1][1] = 1.0f; BT[1][2] = 1.0f; BT[1][3] = 0.0f;
        BT[2][0] = 0.0f; BT[2][1] = -1.0f; BT[2][2] = 1.0f; BT[2][3] = 0.0f;
        BT[3][0] = 0.0f; BT[3][1] = 1.0f; BT[3][2] = 0.0f; BT[3][3] = -1.0f;
        
        // AT matrix
        AT[0][0] = 1.0f; AT[0][1] = 1.0f; AT[0][2] = 1.0f; AT[0][3] = 0.0f;
        AT[1][0] = 0.0f; AT[1][1] = 1.0f; AT[1][2] = -1.0f; AT[1][3] = -1.0f;
    }
    
    __syncthreads();
    
    int tile_x = blockIdx.x * 2;  // Each tile produces 2x2 output
    int tile_y = blockIdx.y * 2;
    
    if (tile_x < out_width && tile_y < out_height) {
        // Load 4x4 input tile
        float d[4][4];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int in_x = tile_x + j - 1; // -1 for padding
                int in_y = tile_y + i - 1;
                
                if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                    d[i][j] = input[in_y * width + in_x];
                } else {
                    d[i][j] = 0.0f;
                }
            }
        }
        
        // Transform input: V = BT * d * B
        float V[4][4] = {0};
        // Simplified computation for demonstration
        // In practice, you'd implement full Winograd transforms
        
        // For now, fall back to direct convolution for this tile
        for (int oy = 0; oy < 2 && tile_y + oy < out_height; oy++) {
            for (int ox = 0; ox < 2 && tile_x + ox < out_width; ox++) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int in_x = tile_x + ox + kx - 1;
                        int in_y = tile_y + oy + ky - 1;
                        
                        if (in_x >= 0 && in_x < width && in_y >= 0 && in_y < height) {
                            sum += input[in_y * width + in_x] * kernel[ky * 3 + kx];
                        }
                    }
                }
                output[(tile_y + oy) * out_width + (tile_x + ox)] = sum;
            }
        }
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

void initialize_random(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void create_gaussian_kernel(float* kernel, int size, float sigma = 1.0f) {
    int center = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int dx = x - center;
            int dy = y - center;
            float value = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            kernel[y * size + x] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

double benchmark_convolution(void(*conv_func)(float*, float*, float*, int, int, int, int, int, int, int),
                            float* d_input, float* d_kernel, float* d_output,
                            int in_h, int in_w, int out_h, int out_w, int k_size,
                            const char* name) {
    CUDA_CHECK(cudaMemset(d_output, 0, out_h * out_w * sizeof(float)));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((out_w + blockSize.x - 1) / blockSize.x,
                  (out_h + blockSize.y - 1) / blockSize.y);
    
    // Warm up
    conv_func<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, in_h, in_w, out_h, out_w, k_size, 1, k_size/2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    const int num_runs = 20;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        conv_func<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, in_h, in_w, out_h, out_w, k_size, 1, k_size/2);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    
    // Calculate throughput (pixels processed per second)
    double pixels_per_sec = (out_h * out_w / (avg_time / 1000.0)) / 1e6;
    
    std::cout << name << ": " << avg_time << " ms (" << pixels_per_sec << " MP/s)" << std::endl;
    
    return avg_time;
}

int main() {
    // Input image dimensions
    const int in_height = 2048;
    const int in_width = 2048;
    const int kernel_size = 5;
    const int stride = 1;
    const int padding = kernel_size / 2;
    
    const int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    std::cout << "=== 2D Convolution Benchmark ===" << std::endl;
    std::cout << "Input: " << in_height << "x" << in_width << std::endl;
    std::cout << "Kernel: " << kernel_size << "x" << kernel_size << std::endl;
    std::cout << "Output: " << out_height << "x" << out_width << std::endl;
    
    // Host memory
    float *h_input = new float[in_height * in_width];
    float *h_kernel = new float[kernel_size * kernel_size];
    float *h_output = new float[out_height * out_width];
    
    initialize_random(h_input, in_height * in_width);
    create_gaussian_kernel(h_kernel, kernel_size, 1.5f);
    
    // Device memory
    float *d_input, *d_kernel, *d_output, *d_temp;
    CUDA_CHECK(cudaMalloc(&d_input, in_height * in_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, out_height * out_width * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, in_height * in_width * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, in_height * in_width * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark different implementations
    double naive_time = benchmark_convolution(naive_conv2d, d_input, d_kernel, d_output,
                                            in_height, in_width, out_height, out_width, kernel_size,
                                            "Naive Convolution");
    
    double shared_time = benchmark_convolution(shared_conv2d, d_input, d_kernel, d_output,
                                             in_height, in_width, out_height, out_width, kernel_size,
                                             "Shared Memory Convolution");
    
    // Separable convolution benchmark
    float *h_kernel_1d = new float[kernel_size];
    for (int i = 0; i < kernel_size; i++) {
        h_kernel_1d[i] = h_kernel[i * kernel_size + kernel_size/2]; // Extract center row
    }
    
    float *d_kernel_1d;
    CUDA_CHECK(cudaMalloc(&d_kernel_1d, kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kernel_1d, h_kernel_1d, kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize((in_width + blockSize.x - 1) / blockSize.x,
                  (in_height + blockSize.y - 1) / blockSize.y);
    
    auto start = std::chrono::high_resolution_clock::now();
    const int num_runs = 20;
    for (int i = 0; i < num_runs; i++) {
        separable_conv_horizontal<<<gridSize, blockSize>>>(d_input, d_temp, d_kernel_1d, in_height, in_width, kernel_size, padding);
        separable_conv_vertical<<<gridSize, blockSize>>>(d_temp, d_output, d_kernel_1d, in_height, in_width, kernel_size, padding);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double separable_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    double separable_pixels_per_sec = (out_height * out_width / (separable_time / 1000.0)) / 1e6;
    std::cout << "Separable Convolution: " << separable_time << " ms (" << separable_pixels_per_sec << " MP/s)" << std::endl;
    
    std::cout << "\nPerformance Comparison:" << std::endl;
    std::cout << "Shared memory speedup: " << naive_time / shared_time << "x" << std::endl;
    std::cout << "Separable speedup: " << naive_time / separable_time << "x" << std::endl;
    
    // Verify correctness by comparing outputs
    float *h_naive = new float[out_height * out_width];
    float *h_shared = new float[out_height * out_width];
    
    // Get naive result
    naive_conv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, in_height, in_width, out_height, out_width, kernel_size, 1, padding);
    CUDA_CHECK(cudaMemcpy(h_naive, d_output, out_height * out_width * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Get shared memory result
    shared_conv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, in_height, in_width, out_height, out_width, kernel_size, 1, padding);
    CUDA_CHECK(cudaMemcpy(h_shared, d_output, out_height * out_width * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < out_height * out_width; i++) {
        float diff = fabsf(h_naive[i] - h_shared[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    
    std::cout << "\nVerification:" << std::endl;
    std::cout << "Max difference between naive and optimized: " << max_diff << std::endl;
    std::cout << (max_diff < 1e-5f ? "PASSED" : "FAILED") << std::endl;
    
    // Memory usage analysis
    size_t input_memory = in_height * in_width * sizeof(float);
    size_t kernel_memory = kernel_size * kernel_size * sizeof(float);
    size_t output_memory = out_height * out_width * sizeof(float);
    size_t total_memory = input_memory + kernel_memory + output_memory;
    
    std::cout << "\nMemory Usage:" << std::endl;
    std::cout << "Input: " << input_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "Kernel: " << kernel_memory / 1024 << " KB" << std::endl;
    std::cout << "Output: " << output_memory / (1024*1024) << " MB" << std::endl;
    std::cout << "Total: " << total_memory / (1024*1024) << " MB" << std::endl;
    
    // Compute operations analysis
    long long ops_per_output = kernel_size * kernel_size * 2; // multiply + add
    long long total_ops = (long long)out_height * out_width * ops_per_output;
    double gops_naive = (total_ops / (naive_time / 1000.0)) / 1e9;
    double gops_shared = (total_ops / (shared_time / 1000.0)) / 1e9;
    
    std::cout << "\nCompute Performance:" << std::endl;
    std::cout << "Total operations: " << total_ops / 1e9 << " GOP" << std::endl;
    std::cout << "Naive: " << gops_naive << " GOPS" << std::endl;
    std::cout << "Shared: " << gops_shared << " GOPS" << std::endl;
    
    // Cleanup
    delete[] h_input; delete[] h_kernel; delete[] h_output; 
    delete[] h_kernel_1d; delete[] h_naive; delete[] h_shared;
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_kernel)); 
    CUDA_CHECK(cudaFree(d_output)); CUDA_CHECK(cudaFree(d_temp)); 
    CUDA_CHECK(cudaFree(d_kernel_1d));
    
    return 0;
}
