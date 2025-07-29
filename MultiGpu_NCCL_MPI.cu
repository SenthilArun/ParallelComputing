#include <nccl.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cmath>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
} while(0)

#define NCCL_CHECK(call) do { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << " - " << ncclGetErrorString(res) << endl; \
        exit(1); \
    } \
} while(0)

// EXTREME computation kernels designed to saturate GPU cores
__global__ void extremeComputeKernel(float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        
        // EXTREME mathematical operations to stress GPU
        for (int i = 0; i < iterations; i++) {
            // Trigonometric operations (expensive)
            val = sinf(val * 1.1f) * cosf(val * 0.9f);
            val = tanf(val * 0.1f) + atanf(val * 2.0f);
            
            // Exponential and logarithmic operations (very expensive)
            val = expf(val * 0.01f) - logf(fabsf(val * 10.0f) + 1.0f);
            
            // Square root and power operations
            val = sqrtf(fabsf(val)) * powf(fabsf(val) + 0.1f, 0.33f);
            
            // More trigonometric stress
            val = sinhf(val * 0.1f) * coshf(val * 0.1f);
            
            // Normalization to prevent overflow
            val = fmaxf(fminf(val, 10.0f), -10.0f);
        }
        
        data[idx] = val;
    }
}

__global__ void massiveMatrixKernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[idy * n + k] * b[k * n + idx];
        }
        c[idy * n + idx] = sum;
    }
}

class ExtremeGPUStressor {
private:
    int num_gpus;
    vector<int> device_ids;
    vector<cudaStream_t> streams;
    vector<cublasHandle_t> cublas_handles;
    vector<curandGenerator_t> curand_gens;
    vector<ncclComm_t> nccl_comms;
    
    // MASSIVE network parameters for maximum memory usage
    int batch_size;
    int input_size;
    int hidden1, hidden2, hidden3, hidden4, hidden5;
    int output_size;
    float learning_rate;
    
    // Multiple large memory buffers per GPU
    vector<float*> d_input, d_output, d_targets;
    vector<float*> d_weights1, d_weights2, d_weights3, d_weights4, d_weights5, d_weights6;
    vector<float*> d_hidden1, d_hidden2, d_hidden3, d_hidden4, d_hidden5;
    vector<float*> d_gradients1, d_gradients2, d_gradients3, d_gradients4, d_gradients5, d_gradients6;
    
    // Additional massive buffers to fill GPU memory
    vector<float*> d_buffer1, d_buffer2, d_buffer3, d_buffer4;
    vector<float*> d_temp_matrices1, d_temp_matrices2, d_temp_matrices3;

public:
    ExtremeGPUStressor() {
        // Get available GPU memory and calculate optimal sizes
        CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        num_gpus = min(num_gpus, 2);
        
        cout << "🔥🔥🔥 EXTREME GPU MEMORY STRESS TEST 🔥🔥🔥" << endl;
        
        // Calculate sizes based on available GPU memory
        calculateOptimalSizes();
        
        device_ids.resize(num_gpus);
        for (int i = 0; i < num_gpus; i++) {
            device_ids[i] = i;
        }
        
        initializeNCCL();
        initializeGPUs();
        allocateMaximumMemory();
        initializeWeights();
        
        cout << "🚀 EXTREME GPU stress test ready!" << endl;
    }
    
    ~ExtremeGPUStressor() {
        cleanup();
    }

    void extremeTrainingStep() {
        cout << "🔥🔥🔥 EXTREME COMPUTATION STEP 🔥🔥🔥" << endl;
        
        // Generate and load massive amounts of data
        vector<float> h_input(batch_size * input_size);
        vector<float> h_targets(batch_size * output_size);
        
        for (int i = 0; i < batch_size * input_size; i++) {
            h_input[i] = sinf(i * 0.001f) * cosf(i * 0.0001f);
        }
        for (int i = 0; i < batch_size * output_size; i++) {
            h_targets[i] = tanf(i * 0.0001f);
        }
        
        // Copy to all GPUs
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            CUDA_CHECK(cudaMemcpyAsync(d_input[i], h_input.data(), 
                                      batch_size * input_size * sizeof(float), 
                                      cudaMemcpyHostToDevice, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(d_targets[i], h_targets.data(), 
                                      batch_size * output_size * sizeof(float), 
                                      cudaMemcpyHostToDevice, streams[i]));
        }
        
        // EXTREME forward pass with massive computations
        cout << "  ⚡ EXTREME forward pass..." << endl;
        extremeForwardPass();
        
        // EXTREME backward pass
        cout << "  ⚡ EXTREME backward pass..." << endl;
        extremeBackwardPass();
        
        // Intensive buffer operations
        cout << "  ⚡ EXTREME buffer operations..." << endl;
        extremeBufferOperations();
        
        // NCCL synchronization
        if (num_gpus > 1) {
            cout << "  ⚡ NCCL gradient synchronization..." << endl;
            allReduceGradients();
        }
        
        // Weight updates with additional computation
        cout << "  ⚡ Weight updates..." << endl;
        updateWeights();
        
        // Synchronize all operations
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
        
        cout << "✅ EXTREME computation step completed!" << endl;
    }
    
    void printExtremeGPUStatus() {
        cout << "\n🔥🔥🔥 EXTREME GPU STATUS 🔥🔥🔥" << endl;
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            size_t free_mem, total_mem;
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            
            size_t used_mem = total_mem - free_mem;
            float mem_percent = (float)used_mem / total_mem * 100.0f;
            
            cout << "  🔥 GPU " << i << " (" << prop.name << "):" << endl;
            cout << "     Memory: " << used_mem / (1024*1024) << "/" 
                 << total_mem / (1024*1024) << " MB (" << fixed << setprecision(1) 
                 << mem_percent << "%)" << endl;
            cout << "     Temperature: Check nvidia-smi for temps!" << endl;
            cout << "     Expected: 95%+ utilization, 80%+ memory, fans at max!" << endl;
        }
        cout << endl;
    }

private:
    void calculateOptimalSizes() {
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            size_t free_mem, total_mem;
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            
            cout << "GPU " << i << " (" << prop.name << "): " 
                 << total_mem/(1024*1024) << " MB total, " 
                 << free_mem/(1024*1024) << " MB free" << endl;
        }
        
        // Set MASSIVE sizes to stress GPUs
        batch_size = 2048;      // Huge batch
        input_size = 4096;      // Large input
        hidden1 = 8192;         // Massive hidden layers
        hidden2 = 6144;
        hidden3 = 4096;
        hidden4 = 2048;
        hidden5 = 1024;
        output_size = 512;
        learning_rate = 0.00001f;
        
        cout << "Network configuration:" << endl;
        cout << "  Batch size: " << batch_size << endl;
        cout << "  Architecture: " << input_size << " → " << hidden1 << " → " 
             << hidden2 << " → " << hidden3 << " → " << hidden4 << " → " 
             << hidden5 << " → " << output_size << endl;
    }
    
    void initializeNCCL() {
        if (num_gpus < 2) return;
        
        ncclUniqueId nccl_id;
        nccl_comms.resize(num_gpus);
        
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            NCCL_CHECK(ncclCommInitRank(&nccl_comms[i], num_gpus, nccl_id, i));
        }
        NCCL_CHECK(ncclGroupEnd());
        
        cout << "✅ NCCL initialized for " << num_gpus << " GPUs" << endl;
    }
    
    void initializeGPUs() {
        streams.resize(num_gpus);
        cublas_handles.resize(num_gpus);
        curand_gens.resize(num_gpus);
        
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            cublasCreate(&cublas_handles[i]);
            cublasSetStream(cublas_handles[i], streams[i]);
            curandCreateGenerator(&curand_gens[i], CURAND_RNG_PSEUDO_DEFAULT);
            curandSetStream(curand_gens[i], streams[i]);
        }
    }
    
    void allocateMaximumMemory() {
        // Resize all vectors
        d_input.resize(num_gpus); d_output.resize(num_gpus); d_targets.resize(num_gpus);
        d_weights1.resize(num_gpus); d_weights2.resize(num_gpus); d_weights3.resize(num_gpus);
        d_weights4.resize(num_gpus); d_weights5.resize(num_gpus); d_weights6.resize(num_gpus);
        d_hidden1.resize(num_gpus); d_hidden2.resize(num_gpus); d_hidden3.resize(num_gpus);
        d_hidden4.resize(num_gpus); d_hidden5.resize(num_gpus);
        d_gradients1.resize(num_gpus); d_gradients2.resize(num_gpus); d_gradients3.resize(num_gpus);
        d_gradients4.resize(num_gpus); d_gradients5.resize(num_gpus); d_gradients6.resize(num_gpus);
        d_buffer1.resize(num_gpus); d_buffer2.resize(num_gpus); d_buffer3.resize(num_gpus); d_buffer4.resize(num_gpus);
        d_temp_matrices1.resize(num_gpus); d_temp_matrices2.resize(num_gpus); d_temp_matrices3.resize(num_gpus);
        
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            cout << "🔥 Allocating MAXIMUM memory on GPU " << i << "..." << endl;
            
            // Core network components
            CUDA_CHECK(cudaMalloc(&d_input[i], batch_size * input_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_output[i], batch_size * output_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_targets[i], batch_size * output_size * sizeof(float)));
            
            // Massive weight matrices
            CUDA_CHECK(cudaMalloc(&d_weights1[i], input_size * hidden1 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_weights2[i], hidden1 * hidden2 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_weights3[i], hidden2 * hidden3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_weights4[i], hidden3 * hidden4 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_weights5[i], hidden4 * hidden5 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_weights6[i], hidden5 * output_size * sizeof(float)));
            
            // Hidden layer activations
            CUDA_CHECK(cudaMalloc(&d_hidden1[i], batch_size * hidden1 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hidden2[i], batch_size * hidden2 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hidden3[i], batch_size * hidden3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hidden4[i], batch_size * hidden4 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_hidden5[i], batch_size * hidden5 * sizeof(float)));
            
            // Gradient matrices
            CUDA_CHECK(cudaMalloc(&d_gradients1[i], input_size * hidden1 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_gradients2[i], hidden1 * hidden2 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_gradients3[i], hidden2 * hidden3 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_gradients4[i], hidden3 * hidden4 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_gradients5[i], hidden4 * hidden5 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_gradients6[i], hidden5 * output_size * sizeof(float)));
            
            // Additional massive buffers to fill remaining memory
            size_t base_buffer_size = 256 * 1024 * 1024;  // 256MB base
            size_t buffer_size = (i == 0) ? base_buffer_size * 4 : base_buffer_size * 2; // Larger for RTX5080
            
            CUDA_CHECK(cudaMalloc(&d_buffer1[i], buffer_size));
            CUDA_CHECK(cudaMalloc(&d_buffer2[i], buffer_size));
            CUDA_CHECK(cudaMalloc(&d_buffer3[i], buffer_size/2));
            CUDA_CHECK(cudaMalloc(&d_buffer4[i], buffer_size/2));
            
            // Massive temporary matrices for computation
            CUDA_CHECK(cudaMalloc(&d_temp_matrices1[i], 2048 * 2048 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_temp_matrices2[i], 2048 * 2048 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_temp_matrices3[i], 2048 * 2048 * sizeof(float)));
            
            // Check actual memory usage
            size_t free_mem, total_mem;
            CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            size_t used_mem = total_mem - free_mem;
            float usage_percent = (float)used_mem / total_mem * 100.0f;
            
            cout << "  📊 GPU " << i << " Memory: " << used_mem/(1024*1024) 
                 << "/" << total_mem/(1024*1024) << " MB (" 
                 << fixed << setprecision(1) << usage_percent << "%)" << endl;
        }
    }
    
    void initializeWeights() {
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            // Initialize all weight matrices with random values
            curandGenerateUniform(curand_gens[i], d_weights1[i], input_size * hidden1);
            curandGenerateUniform(curand_gens[i], d_weights2[i], hidden1 * hidden2);
            curandGenerateUniform(curand_gens[i], d_weights3[i], hidden2 * hidden3);
            curandGenerateUniform(curand_gens[i], d_weights4[i], hidden3 * hidden4);
            curandGenerateUniform(curand_gens[i], d_weights5[i], hidden4 * hidden5);
            curandGenerateUniform(curand_gens[i], d_weights6[i], hidden5 * output_size);
            
            // Fill buffers with random data for intensive computation
            curandGenerateUniform(curand_gens[i], d_buffer1[i], 64*1024*1024);
            curandGenerateUniform(curand_gens[i], d_buffer2[i], 64*1024*1024);
        }
        
        if (num_gpus > 1) {
            broadcastWeights();
        }
    }
    
    void extremeForwardPass() {
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            const float alpha = 1.0f, beta = 0.0f;
            
            // Layer 1 with extreme computation
            cublasSgemm(cublas_handles[i], CUBLAS_OP_N, CUBLAS_OP_N,
                       hidden1, batch_size, input_size,
                       &alpha, d_weights1[i], hidden1,
                       d_input[i], input_size,
                       &beta, d_hidden1[i], hidden1);
            
            // EXTREME computation on activations
            dim3 block(256);
            dim3 grid((batch_size * hidden1 + block.x - 1) / block.x);
            extremeComputeKernel<<<grid, block, 0, streams[i]>>>(d_hidden1[i], batch_size * hidden1, 200);
            
            // Continue through all layers
            cublasSgemm(cublas_handles[i], CUBLAS_OP_N, CUBLAS_OP_N,
                       hidden2, batch_size, hidden1,
                       &alpha, d_weights2[i], hidden2,
                       d_hidden1[i], hidden1,
                       &beta, d_hidden2[i], hidden2);
            
            grid = dim3((batch_size * hidden2 + block.x - 1) / block.x);
            extremeComputeKernel<<<grid, block, 0, streams[i]>>>(d_hidden2[i], batch_size * hidden2, 200);
            
            // Remaining layers...
            cublasSgemm(cublas_handles[i], CUBLAS_OP_N, CUBLAS_OP_N,
                       output_size, batch_size, hidden2,
                       &alpha, d_weights3[i], output_size,
                       d_hidden2[i], hidden2,
                       &beta, d_output[i], output_size);
        }
    }
    
    void extremeBackwardPass() {
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            const float alpha = 1.0f, beta = 0.0f;
            
            // Simplified backward pass with intensive computation
            cublasSgemm(cublas_handles[i], CUBLAS_OP_N, CUBLAS_OP_T,
                       output_size, hidden2, batch_size,
                       &alpha, d_output[i], output_size,
                       d_hidden2[i], hidden2,
                       &beta, d_gradients3[i], output_size);
            
            // Add intensive computation to gradients
            dim3 block(256);
            dim3 grid((hidden2 * output_size + block.x - 1) / block.x);
            extremeComputeKernel<<<grid, block, 0, streams[i]>>>(d_gradients3[i], hidden2 * output_size, 150);
        }
    }
    
    void extremeBufferOperations() {
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            // Perform massive matrix operations on buffers to stress GPU
            dim3 block(16, 16);
            dim3 grid(64, 64);  // Large grid for massive parallel work
            
            massiveMatrixKernel<<<grid, block, 0, streams[i]>>>(
                d_buffer1[i], d_buffer2[i], d_buffer3[i], 1024);
            
            // Additional intensive operations
            dim3 block1d(256);
            dim3 grid1d(4096);
            extremeComputeKernel<<<grid1d, block1d, 0, streams[i]>>>(d_buffer1[i], 64*1024*1024, 300);
            extremeComputeKernel<<<grid1d, block1d, 0, streams[i]>>>(d_buffer2[i], 64*1024*1024, 300);
        }
    }
    
    void allReduceGradients() {
        if (num_gpus < 2) return;
        
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            NCCL_CHECK(ncclAllReduce(d_gradients1[i], d_gradients1[i], 
                                    input_size * hidden1, ncclFloat, ncclSum, 
                                    nccl_comms[i], streams[i]));
            NCCL_CHECK(ncclAllReduce(d_gradients3[i], d_gradients3[i], 
                                    hidden2 * output_size, ncclFloat, ncclSum, 
                                    nccl_comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());
        
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }
    
    void updateWeights() {
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            float neg_lr = -learning_rate;
            
            cublasSaxpy(cublas_handles[i], input_size * hidden1, 
                       &neg_lr, d_gradients1[i], 1, d_weights1[i], 1);
            cublasSaxpy(cublas_handles[i], hidden2 * output_size, 
                       &neg_lr, d_gradients3[i], 1, d_weights3[i], 1);
        }
    }
    
    void broadcastWeights() {
        if (num_gpus < 2) return;
        
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            NCCL_CHECK(ncclBcast(d_weights1[i], input_size * hidden1, ncclFloat, 0, nccl_comms[i], streams[i]));
            NCCL_CHECK(ncclBcast(d_weights3[i], hidden2 * output_size, ncclFloat, 0, nccl_comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());
    }
    
    void cleanup() {
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CHECK(cudaSetDevice(device_ids[i]));
            
            // Free all massive allocations
            if (d_input[i]) cudaFree(d_input[i]);
            if (d_output[i]) cudaFree(d_output[i]);
            if (d_targets[i]) cudaFree(d_targets[i]);
            if (d_weights1[i]) cudaFree(d_weights1[i]);
            if (d_weights2[i]) cudaFree(d_weights2[i]);
            if (d_weights3[i]) cudaFree(d_weights3[i]);
            if (d_weights4[i]) cudaFree(d_weights4[i]);
            if (d_weights5[i]) cudaFree(d_weights5[i]);
            if (d_weights6[i]) cudaFree(d_weights6[i]);
            if (d_hidden1[i]) cudaFree(d_hidden1[i]);
            if (d_hidden2[i]) cudaFree(d_hidden2[i]);
            if (d_hidden3[i]) cudaFree(d_hidden3[i]);
            if (d_hidden4[i]) cudaFree(d_hidden4[i]);
            if (d_hidden5[i]) cudaFree(d_hidden5[i]);
            if (d_gradients1[i]) cudaFree(d_gradients1[i]);
            if (d_gradients2[i]) cudaFree(d_gradients2[i]);
            if (d_gradients3[i]) cudaFree(d_gradients3[i]);
            if (d_gradients4[i]) cudaFree(d_gradients4[i]);
            if (d_gradients5[i]) cudaFree(d_gradients5[i]);
            if (d_gradients6[i]) cudaFree(d_gradients6[i]);
            if (d_buffer1[i]) cudaFree(d_buffer1[i]);
            if (d_buffer2[i]) cudaFree(d_buffer2[i]);
            if (d_buffer3[i]) cudaFree(d_buffer3[i]);
            if (d_buffer4[i]) cudaFree(d_buffer4[i]);
            if (d_temp_matrices1[i]) cudaFree(d_temp_matrices1[i]);
            if (d_temp_matrices2[i]) cudaFree(d_temp_matrices2[i]);
            if (d_temp_matrices3[i]) cudaFree(d_temp_matrices3[i]);
            
            // Cleanup resources
            if (streams[i]) cudaStreamDestroy(streams[i]);
            if (cublas_handles[i]) cublasDestroy(cublas_handles[i]);
            if (curand_gens[i]) curandDestroyGenerator(curand_gens[i]);
            if (i < nccl_comms.size() && nccl_comms[i]) {
                ncclCommDestroy(nccl_comms[i]);
            }
        }
    }
};

int main() {
    try {
        cout << "🚀🚀🚀 LAUNCHING EXTREME GPU STRESS TEST 🚀🚀🚀" << endl;
        cout << "WARNING: This will push your GPUs to their absolute limits!" << endl;
        cout << "Monitor temperatures with: watch -n 0.5 nvidia-smi" << endl;
        cout << "Press Ctrl+C if temperatures exceed 85°C" << endl << endl;
        
        // Initialize the extreme GPU stressor
        ExtremeGPUStressor stressor;
        
        cout << "\n🔥 Starting extreme training loop..." << endl;
        
        // Run intensive training loop
        for (int epoch = 0; epoch < 30; epoch++) {
            cout << "\n⚡⚡⚡ EXTREME EPOCH " << epoch << " ⚡⚡⚡" << endl;
            
            auto start = chrono::high_resolution_clock::now();
            
            // Perform extreme training step
            stressor.extremeTrainingStep();
            
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            
            cout << "🏁 EXTREME EPOCH " << epoch << " completed in " << duration.count() << " ms" << endl;
            
            // Show GPU status every few epochs
            if (epoch % 3 == 0) {
                stressor.printExtremeGPUStatus();
            }
            
            // Small delay to allow sustained high utilization
            this_thread::sleep_for(chrono::milliseconds(200));
            
            cout << "🌡️  Check nvidia-smi now - both GPUs should be at 95%+ utilization!" << endl;
        }
        
        cout << "\n🎉🎉🎉 EXTREME GPU STRESS TEST COMPLETED! 🎉🎉🎉" << endl;
        cout << "Your GPUs should have been running at maximum capacity!" << endl;
        
        // Final status report
        stressor.printExtremeGPUStatus();
        
    } catch (const exception& e) {
        cerr << "❌ Error: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}
