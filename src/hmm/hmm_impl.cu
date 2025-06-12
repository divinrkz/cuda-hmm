#include "../../include/hmm.hpp"
#include "../../include/hmm_cuda.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <float.h>
#include <cmath>
#include <iostream>
#include <sstream>

#define cudaCheck(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

struct log_transform {
    __device__ float operator()(float x) const {
        return logf(fmaxf(x, 1e-40f));
    }
};

struct normalize_transform {
    float sum;
    normalize_transform(float s) : sum(s) {}
    __device__ float operator()(float x) const {
        return (sum > 0) ? x / sum : 0.0f;
    }
};

struct init_alpha_functor {
    float* start_p;
    float* emit_p;
    int* obs;
    int M;
    
    init_alpha_functor(float* s, float* e, int* o, int m) : start_p(s), emit_p(e), obs(o), M(m) {}
    
    __device__ float operator()(int i) const {
        return start_p[i] * emit_p[i * M + obs[0]];
    }
};

struct init_viterbi_functor {
    float* log_start_p;
    float* log_emit_p;
    int* obs;
    int M;
    
    init_viterbi_functor(float* s, float* e, int* o, int m) : log_start_p(s), log_emit_p(e), obs(o), M(m) {}
    
    __device__ float operator()(int i) const {
        return log_start_p[i] + log_emit_p[i * M + obs[0]];
    }
};



void HMMCuda::initializeDeviceMemory(int max_T, int num_states, int num_observations) {
    max_T_allocated = max_T;

    cudaCheck(cudaMalloc(&d_alpha, max_T * num_states * sizeof(float)));
    cudaCheck(cudaMalloc(&d_beta, max_T * num_states * sizeof(float)));
    cudaCheck(cudaMalloc(&d_gamma, max_T * num_states * sizeof(float)));
    cudaCheck(cudaMalloc(&d_xi, max_T * num_states * num_states * sizeof(float)));
    
    cudaCheck(cudaMalloc(&d_V_curr, num_states * sizeof(float)));
    cudaCheck(cudaMalloc(&d_V_prev, num_states * sizeof(float)));
    
    cudaCheck(cudaMalloc(&d_log_trans_p, num_states * num_states * sizeof(float)));
    cudaCheck(cudaMalloc(&d_log_emit_p, num_states * num_observations * sizeof(float)));
    cudaCheck(cudaMalloc(&d_log_start_p, num_states * sizeof(float)));
    
    cudaCheck(cudaMalloc(&d_path, max_T * num_states * sizeof(int)));
    cudaCheck(cudaMalloc(&d_obs, max_T * sizeof(int)));
    
    h_alpha = new float*[max_T];
    h_beta = new float*[max_T];
    for (int t = 0; t < max_T; t++) {
        h_alpha[t] = new float[num_states];
        h_beta[t] = new float[num_states];
    }
}

void HMMCuda::freeDeviceMemory() {
    cudaCheck(cudaFree(d_alpha));
    cudaCheck(cudaFree(d_beta));
    cudaCheck(cudaFree(d_gamma));
    cudaCheck(cudaFree(d_xi));
    cudaCheck(cudaFree(d_V_curr));
    cudaCheck(cudaFree(d_V_prev));
    cudaCheck(cudaFree(d_log_trans_p));
    cudaCheck(cudaFree(d_log_emit_p));
    cudaCheck(cudaFree(d_log_start_p));
    cudaCheck(cudaFree(d_path));
    cudaCheck(cudaFree(d_obs));
    
    if (h_alpha) {
        for (int t = 0; t < max_T_allocated; t++) {
            delete[] h_alpha[t];
            delete[] h_beta[t];
        }
        delete[] h_alpha;
        delete[] h_beta;
    }
}

void HMMCuda::convertToLogSpace(float* probs, float* log_probs, int size) {
    thrust::device_ptr<float> d_probs(probs);
    thrust::device_ptr<float> d_log_probs(log_probs);
    
    thrust::transform(d_probs, d_probs + size, d_log_probs, log_transform());
}

void HMMCuda::normalizeArray(float* array, int size) {
    thrust::device_ptr<float> d_array(array);
    float sum = thrust::reduce(d_array, d_array + size, 0.0f, thrust::plus<float>());
    
    if (sum > 0) {
        thrust::transform(d_array, d_array + size, d_array, normalize_transform(sum));
    }
}

HMMCuda::HMMCuda(int num_states, int num_observations) : IHMM(num_states, num_observations) {
    initializeDeviceMemory(10000, num_states, num_observations);
}

HMMCuda::~HMMCuda() {
    freeDeviceMemory();
}


// Forward algorithm kernel
__global__ void forwardKernel(float* alpha_curr, float* alpha_prev, float* trans_p, 
                            float* emit_p, int* obs, int N, int M, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += alpha_prev[j] * trans_p[j * N + i];
        }
        alpha_curr[i] = sum * emit_p[i * M + obs[t]];
    }
}

// Backward algorithm kernel
__global__ void backwardKernel(float* beta_curr, float* beta_next, float* trans_p,
                             float* emit_p, int* obs, int N, int M, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += trans_p[i * N + j] * emit_p[j * M + obs[t+1]] * beta_next[j];
        }
        beta_curr[i] = sum;
    }
}

__global__ void viterbiKernelOptimized(
    float* V_curr, float* V_prev, int* path,
    float* log_trans_p, float* log_emit_p, int* obs,
    int N, int M, int t) {
    
    extern __shared__ float sdata[];
    float* s_V_prev = sdata;
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load previous V values into shared memory
    if (tid < N && tid < blockDim.x) {
        s_V_prev[tid] = (tid < N) ? V_prev[tid] : -FLT_MAX;
    }
    __syncthreads();
    
    if (i < N) {
        float max_val = -FLT_MAX;
        int max_state = 0;
        
        // Find maximum over all previous states
        for (int j = 0; j < N; j++) {
            float val;
            if (j < blockDim.x) {
                val = s_V_prev[j] + log_trans_p[j * N + i];
            } else {
                val = V_prev[j] + log_trans_p[j * N + i];
            }
            
            if (val > max_val) {
                max_val = val;
                max_state = j;
            }
        }
        
        // Add emission probability
        max_val += log_emit_p[i * M + obs[t]];
        
        V_curr[i] = max_val;
        path[t * N + i] = max_state;
    }
}

// Baum-Welch kernel for computing gamma and xi (E-step)
__global__ void baumWelchEStepKernel(float* gamma, float* xi, float* alpha, float* beta,
                                    float* trans_p, float* emit_p, int* obs, 
                                    int T, int N, int M, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        // gamma[t][i] = alpha[t][i] * beta[t][i]
        gamma[t * N + i] = alpha[t * N + i] * beta[t * N + i];
        
        // xi[t][i][j] for all j
        if (t < T - 1) {
            for (int j = 0; j < N; j++) {
                float xi_val = alpha[t * N + i] * trans_p[i * N + j] * 
                              emit_p[j * M + obs[t+1]] * beta[(t+1) * N + j];
                xi[t * N * N + i * N + j] = xi_val;
            }
        }
    }
}

// Baum-Welch kernel for updating parameters (M-step)
__global__ void baumWelchMStepKernel(float* new_trans_p, float* new_emit_p,
                                    float* gamma, float* xi, int* obs,
                                    int T, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        float xi_sum = 0.0f;
        float gamma_sum = 0.0f;
        
        for (int t = 0; t < T - 1; t++) {
            xi_sum += xi[t * N * N + i * N + j];
            gamma_sum += gamma[t * N + i];
        }
        
        new_trans_p[i * N + j] = (gamma_sum > 0) ? xi_sum / gamma_sum : 0.0f;
    }
    
    if (i < N && j < M) {
        float obs_sum = 0.0f;
        float gamma_sum = 0.0f;
        
        for (int t = 0; t < T; t++) {
            gamma_sum += gamma[t * N + i];
            if (obs[t] == j) {
                obs_sum += gamma[t * N + i];
            }
        }
        new_emit_p[i * M + j] = (gamma_sum > 0) ? obs_sum / gamma_sum : 0.0f;
    }
}



float **HMMCuda::forward(float *obs, int *states, float *start_p, float *trans_p, 
               float *emit_p, int T, int N, int M) {
    
    thrust::host_vector<int> h_obs(T);
    for (int i = 0; i < T; i++) {
        h_obs[i] = static_cast<int>(obs[i]);
    }
    thrust::device_vector<int> d_obs_vec = h_obs;
    cudaCheck(cudaMemcpy(d_obs, thrust::raw_pointer_cast(d_obs_vec.data()), 
                       T * sizeof(int), cudaMemcpyDeviceToDevice));
    
    float *d_trans_p_temp, *d_emit_p_temp, *d_start_p_temp;
    cudaCheck(cudaMalloc(&d_trans_p_temp, N * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_emit_p_temp, N * M * sizeof(float)));
    cudaCheck(cudaMalloc(&d_start_p_temp, N * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_trans_p_temp, trans_p, N * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_emit_p_temp, emit_p, N * M * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_start_p_temp, start_p, N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(min(256, N));
    dim3 grid((N + block.x - 1) / block.x);
    
    thrust::device_ptr<float> d_alpha_ptr(d_alpha);
    
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(N),
        d_alpha_ptr,
        init_alpha_functor(d_start_p_temp, d_emit_p_temp, d_obs, M)
    );
    
    for (int t = 1; t < T; t++) {
        forwardKernel<<<grid, block>>>(
            d_alpha + t * N, d_alpha + (t-1) * N,
            d_trans_p_temp, d_emit_p_temp, d_obs, N, M, t);
        cudaCheck(cudaDeviceSynchronize());
        
        normalizeArray(d_alpha + t * N, N);
    }
    
    for (int t = 0; t < T; t++) {
        cudaCheck(cudaMemcpy(h_alpha[t], d_alpha + t * N, 
                           N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    cudaCheck(cudaFree(d_trans_p_temp));
    cudaCheck(cudaFree(d_emit_p_temp));
    cudaCheck(cudaFree(d_start_p_temp));
    
    return h_alpha;
}

float **HMMCuda::backward(float *obs, int *states, float *start_p, float *trans_p,
                float *emit_p, int T, int N, int M) {
    
    thrust::host_vector<int> h_obs(T);
    for (int i = 0; i < T; i++) {
        h_obs[i] = static_cast<int>(obs[i]);
    }
    thrust::device_vector<int> d_obs_vec = h_obs;
    cudaCheck(cudaMemcpy(d_obs, thrust::raw_pointer_cast(d_obs_vec.data()), 
                       T * sizeof(int), cudaMemcpyDeviceToDevice));
    
    float *d_trans_p_temp, *d_emit_p_temp;
    cudaCheck(cudaMalloc(&d_trans_p_temp, N * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_emit_p_temp, N * M * sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_trans_p_temp, trans_p, N * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_emit_p_temp, emit_p, N * M * sizeof(float), cudaMemcpyHostToDevice));
    
    thrust::device_ptr<float> d_beta_ptr(d_beta + (T-1) * N);
    thrust::fill(d_beta_ptr, d_beta_ptr + N, 1.0f);
    
    dim3 block(min(256, N));
    dim3 grid((N + block.x - 1) / block.x);
    
    for (int t = T - 2; t >= 0; t--) {
        backwardKernel<<<grid, block>>>(
            d_beta + t * N, d_beta + (t+1) * N,
            d_trans_p_temp, d_emit_p_temp, d_obs, N, M, t);
        cudaCheck(cudaDeviceSynchronize());
        
        normalizeArray(d_beta + t * N, N);
    }
    
    for (int t = 0; t < T; t++) {
        cudaCheck(cudaMemcpy(h_beta[t], d_beta + t * N, 
                           N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    cudaCheck(cudaFree(d_trans_p_temp));
    cudaCheck(cudaFree(d_emit_p_temp));
    
    return h_beta;
}

std::string HMMCuda::viterbi(float *obs, int *states, float *start_p, float *trans_p,
                   float *emit_p, int T, int N, int M) {
    
    thrust::host_vector<int> h_obs(T);
    for (int i = 0; i < T; i++) {
        h_obs[i] = static_cast<int>(obs[i]);
    }
    thrust::device_vector<int> d_obs_vec = h_obs;
    cudaCheck(cudaMemcpy(d_obs, thrust::raw_pointer_cast(d_obs_vec.data()), 
                       T * sizeof(int), cudaMemcpyDeviceToDevice));
    
    thrust::device_vector<float> d_trans_vec(N * N), d_emit_vec(N * M), d_start_vec(N);
    cudaCheck(cudaMemcpy(thrust::raw_pointer_cast(d_trans_vec.data()), 
                       trans_p, N * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(thrust::raw_pointer_cast(d_emit_vec.data()), 
                       emit_p, N * M * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(thrust::raw_pointer_cast(d_start_vec.data()), 
                       start_p, N * sizeof(float), cudaMemcpyHostToDevice));
    
    convertToLogSpace(thrust::raw_pointer_cast(d_trans_vec.data()), d_log_trans_p, N * N);
    convertToLogSpace(thrust::raw_pointer_cast(d_emit_vec.data()), d_log_emit_p, N * M);
    convertToLogSpace(thrust::raw_pointer_cast(d_start_vec.data()), d_log_start_p, N);
    
    thrust::device_ptr<float> d_V_prev_ptr(d_V_prev);
    
    thrust::transform(
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(N),
        d_V_prev_ptr,
        init_viterbi_functor(d_log_start_p, d_log_emit_p, d_obs, M)
    );
    
    int threadsPerBlock = min(256, N);
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = min(N, threadsPerBlock) * sizeof(float);
    
    for (int t = 1; t < T; t++) {
        viterbiKernelOptimized<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_V_curr, d_V_prev, d_path, d_log_trans_p, d_log_emit_p, d_obs,
            N, M, t);
        cudaCheck(cudaDeviceSynchronize());
        
        float* temp = d_V_curr;
        d_V_curr = d_V_prev;
        d_V_prev = temp;
    }
    
    thrust::device_ptr<float> d_final_V(d_V_prev);
    auto max_iter = thrust::max_element(d_final_V, d_final_V + N);
    int final_state = max_iter - d_final_V;
    
    thrust::host_vector<int> h_path(T * N);
    cudaCheck(cudaMemcpy(h_path.data(), d_path, 
                       T * N * sizeof(int), cudaMemcpyDeviceToHost));
    
    thrust::host_vector<int> best_path(T);
    best_path[T-1] = final_state;
    
    for (int t = T-2; t >= 0; t--) {
        best_path[t] = h_path[(t+1) * N + best_path[t+1]];
    }
    
    std::ostringstream result;
    for (int t = 0; t < T; t++) {
        result << best_path[t];
        if (t < T-1) result << ""; 
    }
    
    return result.str();
}


void HMMCuda::baum_welch(float *obs, int *states, float *start_p, float *trans_p,
                float *emit_p, int T, int N, int M, int N_iters) {
    
    for (int iter = 0; iter < N_iters; iter++) {
        forward(obs, states, start_p, trans_p, emit_p, T, N, M);
        backward(obs, states, start_p, trans_p, emit_p, T, N, M);
        
        thrust::host_vector<int> h_obs(T);
        for (int i = 0; i < T; i++) {
            h_obs[i] = static_cast<int>(obs[i]);
        }
        thrust::device_vector<int> d_obs_vec = h_obs;
        cudaCheck(cudaMemcpy(d_obs, thrust::raw_pointer_cast(d_obs_vec.data()), 
                           T * sizeof(int), cudaMemcpyDeviceToDevice));
        
        float *d_trans_p_temp, *d_emit_p_temp;
        cudaCheck(cudaMalloc(&d_trans_p_temp, N * N * sizeof(float)));
        cudaCheck(cudaMalloc(&d_emit_p_temp, N * M * sizeof(float)));
        
        cudaCheck(cudaMemcpy(d_trans_p_temp, trans_p, N * N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_emit_p_temp, emit_p, N * M * sizeof(float), cudaMemcpyHostToDevice));
        
        dim3 block1(min(256, N));
        dim3 grid1((N + block1.x - 1) / block1.x);
        
        for (int t = 0; t < T; t++) {
            baumWelchEStepKernel<<<grid1, block1>>>(
                d_gamma, d_xi, d_alpha, d_beta,
                d_trans_p_temp, d_emit_p_temp, d_obs,
                T, N, M, t);
        }
        cudaCheck(cudaDeviceSynchronize());
        
        float *d_new_trans_p, *d_new_emit_p;
        cudaCheck(cudaMalloc(&d_new_trans_p, N * N * sizeof(float)));
        cudaCheck(cudaMalloc(&d_new_emit_p, N * M * sizeof(float)));
        
        dim3 block2(16, 16);
        dim3 grid2((N + block2.x - 1) / block2.x, (max(N, M) + block2.y - 1) / block2.y);
        
        baumWelchMStepKernel<<<grid2, block2>>>(
            d_new_trans_p, d_new_emit_p, d_gamma, d_xi, d_obs,
            T, N, M);
        cudaCheck(cudaDeviceSynchronize());
        
        cudaCheck(cudaMemcpy(trans_p, d_new_trans_p, N * N * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(emit_p, d_new_emit_p, N * M * sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaCheck(cudaFree(d_trans_p_temp));
        cudaCheck(cudaFree(d_emit_p_temp));
        cudaCheck(cudaFree(d_new_trans_p));
        cudaCheck(cudaFree(d_new_emit_p));
    }
}

void HMMCuda::forward_backward(float *obs, int *states, float *start_p, float *trans_p,
                     float *emit_p, int T, int N, int M) {
    forward(obs, states, start_p, trans_p, emit_p, T, N, M);
    backward(obs, states, start_p, trans_p, emit_p, T, N, M);
}