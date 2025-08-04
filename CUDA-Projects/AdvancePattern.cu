// advanced_kernels_fixed.cu - Advanced GPU algorithms and optimizations (Fixed)
// Demonstrates parallel reduction, matrix transpose, and memory coalescing

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <cfloat>      // For FLT_MAX
#include <climits>     // For additional constants

// Optimized parallel reduction using warp shuffle
__global__ void warp_reduce_sum(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Final warp reduction using shuffle
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// Matrix transpose with coalesced memory access
__global__ void coalesced_transpose(float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][33]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Coalesced read from input
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // Coalesced write to output
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Vectorized transpose using float4
__global__ void vectorized_transpose(float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][33];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 8 + threadIdx.y;
    
    // Load 4 elements per thread using float4 (with bounds checking)
    if (x < cols && y * 4 + 3 < rows) {
        float4 val = reinterpret_cast<float4*>(&input[y * 4 * cols + x * 4])[0];
        tile[threadIdx.y * 4][threadIdx.x] = val.x;
        tile[threadIdx.y * 4 + 1][threadIdx.x] = val.y;
        tile[threadIdx.y * 4 + 2][threadIdx.x] = val.z;
        tile[threadIdx.y * 4 + 3][threadIdx.x] = val.w;
    }
    
    __syncthreads();
    
    // Transpose coordinates
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 8 + threadIdx.y;
    
    // Store 4 elements using float4 (with bounds checking)
    if (x < rows && y * 4 + 3 < cols) {
        float4 val;
        val.x = tile[threadIdx.x][threadIdx.y * 4];
        val.y = tile[threadIdx.x][threadIdx.y * 4 + 1];
        val.z = tile[threadIdx.x][threadIdx.y * 4 + 2];
        val.w = tile[threadIdx.x][threadIdx.y * 4 + 3];
        reinterpret_cast<float4*>(&output[y * 4 * rows + x * 4])[0] = val;
    }
}

// Segmented reduction for variable-length sequences
__global__ void segmented_reduction(float* input, int* segment_offsets, float* output, int num_segments) {
    extern __shared__ float sdata[];
    
    int segment_id = blockIdx.x;
    if (segment_id >= num_segments) return;
    
    int start = segment_offsets[segment_id];
    int end = segment_offsets[segment_id + 1];
    int length = end - start;
    
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    // Each thread processes multiple elements
    for (int i = tid; i < length; i += blockDim.x) {
        sum += input[start + i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Block-wise reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[segment_id] = sdata[0];
    }
}

// Scan (prefix sum) implementation
__global__ void inclusive_scan(float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    
    // Load input into shared memory with bounds checking
    temp[pout * blockDim.x + tid] = (tid < n) ? input[tid] : 0.0f;
    
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        __syncthreads();
        
        if (tid >= offset) {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid] + temp[pin * blockDim.x + tid - offset];
        } else {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid];
        }
    }
    
    __syncthreads();
    if (tid < n) {
        output[tid] = temp[pout * blockDim.x + tid];
    }
}

// Histogram computation with shared memory atomics
__global__ void histogram_shared_atomics(float* input, int* histogram, int n, int num_bins, float min_val, float max_val) {
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared histogram
    for (int bin = tid; bin < num_bins; bin += blockDim.x) {
        shared_hist[bin] = 0;
    }
    __syncthreads();
    
    // Compute histogram in shared memory
    if (i < n) {
        float val = input[i];
        int bin = (int)((val - min_val) / (max_val - min_val) * num_bins);
        bin = min(max(bin, 0), num_bins - 1);
        atomicAdd(&shared_hist[bin], 1);
    }
    __syncthreads();
    
    // Add to global histogram
    for (int bin = tid; bin < num_bins; bin += blockDim.x) {
        atomicAdd(&histogram[bin], shared_hist[bin]);
    }
}

// Bitonic sort for small arrays (fixed FLT_MAX issue)
__global__ void bitonic_sort(float* data, int n) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory (use large value instead of FLT_MAX)
    shared_data[tid] = (i < n) ? data[i] : 1e30f;  // Use large float instead of FLT_MAX
    __syncthreads();
    
    // Bitonic sort
    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            
            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                if ((shared_data[tid] > shared_data[ixj]) == ascending) {
                    float temp = shared_data[tid];
                    shared_data[tid] = shared_data[ixj];
                    shared_data[ixj] = temp;
                }
            }
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (i < n) {
        data[i] = shared_data[tid];
    }
}

// Cooperative groups reduction (simplified version)
__global__ void cooperative_reduction(float* input, float* output, int n) {
    namespace cg = cooperative_groups;
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    block.sync();
    
    // Reduction using cooperative groups
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        block.sync();
    }
    
    // Warp-level reduction (manual sum instead of cg::reduce which may not be available)
    if (warp.meta_group_rank() == 0) {
        float val = sdata[warp.thread_rank()];
        
        // Manual warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            val += warp.shfl_down(val, offset);
        }
        
        if (warp.thread_rank() == 0) {
            output[blockIdx.x] = val;
        }
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

void initialize_random(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

double benchmark_reduction(float* d_input, float* d_output, int n, const char* name,
                          void(*kernel)(float*, float*, int)) {
    int num_blocks = (n + 255) / 256;
    
    const int num_runs = 50;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        kernel<<<num_blocks, 256, 256 * sizeof(float)>>>(d_input, d_output, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    double bandwidth = (n * sizeof(float) / (avg_time / 1000.0)) / 1e9;
    
    std::cout << name << ": " << avg_time << " ms (" << bandwidth << " GB/s)" << std::endl;
    return avg_time;
}

double benchmark_transpose(float* d_input, float* d_output, int rows, int cols, const char* name,
                          void(*kernel)(float*, float*, int, int)) {
    dim3 blockSize(32, 8);
    dim3 gridSize((cols + 31) / 32, (rows + 31) / 32);
    
    const int num_runs = 50;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; i++) {
        kernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double avg_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    double bandwidth = (2 * rows * cols * sizeof(float) / (avg_time / 1000.0)) / 1e9;
    
    std::cout << name << ": " << avg_time << " ms (" << bandwidth << " GB/s)" << std::endl;
    return avg_time;
}

int main() {
    std::cout << "=== Advanced CUDA Kernels Benchmark ===" << std::endl;
    
    // Test reduction algorithms
    const int n = 16 * 1024 * 1024;
    float *h_input = new float[n];
    float *h_output = new float[(n + 255) / 256];
    
    initialize_random(h_input, n);
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, ((n + 255) / 256) * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "\n--- Reduction Benchmarks (16M elements) ---" << std::endl;
    benchmark_reduction(d_input, d_output, n, "Warp Shuffle Reduction", warp_reduce_sum);
    benchmark_reduction(d_input, d_output, n, "Cooperative Groups Reduction", cooperative_reduction);
    
    // Compare with CUB
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    float* d_sum;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_sum, n);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    
    auto start = std::chrono::high_resolution_clock::now();
    const int num_runs = 50;
    for (int i = 0; i < num_runs; i++) {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_input, d_sum, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double cub_time = std::chrono::duration<double, std::milli>(end - start).count() / num_runs;
    double cub_bandwidth = (n * sizeof(float) / (cub_time / 1000.0)) / 1e9;
    std::cout << "CUB Reduction: " << cub_time << " ms (" << cub_bandwidth << " GB/s)" << std::endl;
    
    // Test matrix transpose
    const int rows = 4096, cols = 4096;
    float *h_matrix = new float[rows * cols];
    float *h_transposed = new float[rows * cols];
    
    initialize_random(h_matrix, rows * cols);
    
    float *d_matrix, *d_transposed;
    CUDA_CHECK(cudaMalloc(&d_matrix, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_transposed, rows * cols * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "\n--- Transpose Benchmarks (4096x4096) ---" << std::endl;
    benchmark_transpose(d_matrix, d_transposed, rows, cols, "Coalesced Transpose", coalesced_transpose);
    
    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_transposed, d_transposed, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < std::min(rows, 100) && correct; i++) {
        for (int j = 0; j < std::min(cols, 100) && correct; j++) {
            if (fabsf(h_matrix[i * cols + j] - h_transposed[j * rows + i]) > 1e-6f) {
                correct = false;
            }
        }
    }
    
    std::cout << "Transpose correctness: " << (correct ? "PASSED" : "FAILED") << std::endl;
    
    // Test histogram
    const int num_bins = 256;
    int *h_histogram = new int[num_bins];
    int *d_histogram;
    CUDA_CHECK(cudaMalloc(&d_histogram, num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));
    
    dim3 hist_block(256);
    dim3 hist_grid((n + 255) / 256);
    size_t hist_shared_size = num_bins * sizeof(int);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; i++) {
        CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * sizeof(int)));
        histogram_shared_atomics<<<hist_grid, hist_block, hist_shared_size>>>(
            d_input, d_histogram, n, num_bins, 0.0f, 100.0f);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double hist_time = std::chrono::duration<double, std::milli>(end - start).count() / 20.0;
    std::cout << "\n--- Histogram (256 bins, 16M elements) ---" << std::endl;
    std::cout << "Shared Memory Atomics: " << hist_time << " ms" << std::endl;
    
    // Test sorting (smaller array for bitonic sort)
    const int sort_n = 1024;
    float *h_sort_data = new float[sort_n];
    float *d_sort_data;
    
    initialize_random(h_sort_data, sort_n);
    
    CUDA_CHECK(cudaMalloc(&d_sort_data, sort_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_sort_data, h_sort_data, sort_n * sizeof(float), cudaMemcpyHostToDevice));
    
    start = std::chrono::high_resolution_clock::now();
    bitonic_sort<<<1, sort_n, sort_n * sizeof(float)>>>(d_sort_data, sort_n);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double sort_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "\n--- Sorting (1024 elements) ---" << std::endl;
    std::cout << "Bitonic Sort: " << sort_time << " ms" << std::endl;
    
    // Verify sort correctness
    float *h_sorted = new float[sort_n];
    CUDA_CHECK(cudaMemcpy(h_sorted, d_sort_data, sort_n * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool sorted = true;
    for (int i = 1; i < sort_n; i++) {
        if (h_sorted[i] < h_sorted[i-1]) {
            sorted = false;
            break;
        }
    }
    
    std::cout << "Sort correctness: " << (sorted ? "PASSED" : "FAILED") << std::endl;
    
    // Performance summary
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Best reduction bandwidth: " << cub_bandwidth << " GB/s (CUB)" << std::endl;
    
    // Cleanup
    delete[] h_input; delete[] h_output; delete[] h_matrix; delete[] h_transposed;
    delete[] h_histogram; delete[] h_sort_data; delete[] h_sorted;
    CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_matrix)); CUDA_CHECK(cudaFree(d_transposed));
    CUDA_CHECK(cudaFree(d_histogram)); CUDA_CHECK(cudaFree(d_sort_data));
    CUDA_CHECK(cudaFree(d_temp_storage)); CUDA_CHECK(cudaFree(d_sum));
    
    return 0;
}
