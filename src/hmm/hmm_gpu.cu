// File: hmm_gpu.cu
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <limits>
#include <numeric>
#include "hmm_gpu.cuh"

// Check CUDA errors and exit if any occurred
#define CHECK_CUDA_ERROR(err)                                                                              \
    if (err != cudaSuccess)                                                                                \
    {                                                                                                      \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                                \
    }

// Initialize alpha values and scaling factor for t=0
__global__ void kernel_init_alpha_scaled(float *d_alpha, float *d_scaling, const float *d_pi,
                                       const float *d_B, int obs0, int N, int M) 
{
    int i = threadIdx.x;
    if (i >= N)
        return;

    d_alpha[i] = d_pi[i] * d_B[i * M + obs0];

    extern __shared__ float sdata[];
    sdata[i] = d_alpha[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s)
            sdata[i] += sdata[i + s];
        __syncthreads();
    }

    if (i == 0) {
        d_scaling[0] = (sdata[0] > 1e-35f) ? 1.0f / sdata[0] : 0.0f;
    }
    __syncthreads();

    d_alpha[i] *= d_scaling[0];
}

// Forward step with scaling for numerical stability
__global__ void kernel_forward_step_scaled(float *d_alpha_t, float *d_scaling_t, const float *d_alpha_t_minus_1,
                                         const float *d_A, const float *d_B, int obs_t, int N, int M)
{
    int i = threadIdx.x;
    if (i >= N)
        return;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += d_alpha_t_minus_1[j] * d_A[j * N + i];
    }
    d_alpha_t[i] = sum * d_B[i * M + obs_t];

    extern __shared__ float sdata[];
    sdata[i] = d_alpha_t[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (i < s)
            sdata[i] += sdata[i + s];
        __syncthreads();
    }

    if (i == 0) {
        *d_scaling_t = (sdata[0] > 1e-35f) ? 1.0f / sdata[0] : 0.0f;
    }
    __syncthreads();

    d_alpha_t[i] *= (*d_scaling_t);
}

// Backward step with scaling for numerical stability
__global__ void kernel_backward_step_scaled(float *d_beta_t, const float *d_beta_t_plus_1,
                                          const float *d_A, const float *d_B, int obs_t1, float scaling_factor, int N, int M)
{
    int i = threadIdx.x;
    if (i >= N)
        return;

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        sum += d_A[i * N + j] * d_B[j * M + obs_t1] * d_beta_t_plus_1[j];
    }
    d_beta_t[i] = sum * scaling_factor;
}

// Expectation step for Baum-Welch algorithm
__global__ void kernel_expectation_step(
    float *d_A_numer, float *d_A_denom, float *d_O_numer, float *d_O_denom,
    const float *d_alpha, const float *d_beta, const float *d_A, const float *d_B,
    const int *d_obs, int N, int M, int T)
{
    int t = blockIdx.x + 1;
    if (t > T)
        return;

    extern __shared__ float s_buffer[];

    // Calculate gamma denominator
    float my_gamma_denom_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        my_gamma_denom_sum += d_alpha[t * N + i] * d_beta[t * N + i];
    }
    s_buffer[threadIdx.x] = my_gamma_denom_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            s_buffer[threadIdx.x] += s_buffer[threadIdx.x + s];
        __syncthreads();
    }
    float gamma_denom = s_buffer[0];

    // Update accumulators based on gamma
    if (gamma_denom > 1e-35f) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            float gamma_val = (d_alpha[t * N + i] * d_beta[t * N + i]) / gamma_denom;
            int obs_idx = d_obs[t - 1];
            atomicAdd(&d_O_numer[i * M + obs_idx], gamma_val);
            atomicAdd(&d_O_denom[i], gamma_val);
            if (t != T) {
                atomicAdd(&d_A_denom[i], gamma_val);
            }
        }
    }
    __syncthreads();

    // Calculate xi for transition probabilities
    if (t < T) {
        float my_xi_denom_sum = 0.0f;
        int obs_t = d_obs[t];
        for (int flat_idx = threadIdx.x; flat_idx < N * N; flat_idx += blockDim.x) {
            int i = flat_idx / N;
            int j = flat_idx % N;
            my_xi_denom_sum += d_alpha[t * N + i] * d_A[i * N + j] * d_B[j * M + obs_t] * d_beta[(t + 1) * N + j];
        }
        s_buffer[threadIdx.x] = my_xi_denom_sum;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                s_buffer[threadIdx.x] += s_buffer[threadIdx.x + s];
            __syncthreads();
        }
        float xi_denom = s_buffer[0];

        if (xi_denom > 1e-35f) {
            for (int flat_idx = threadIdx.x; flat_idx < N * N; flat_idx += blockDim.x) {
                int i = flat_idx / N;
                int j = flat_idx % N;
                float xi_val = (d_alpha[t * N + i] * d_A[i * N + j] * d_B[j * M + obs_t] * d_beta[(t + 1) * N + j]) / xi_denom;
                atomicAdd(&d_A_numer[i * N + j], xi_val);
            }
        }
    }
}

// Maximization step for Baum-Welch algorithm
__global__ void kernel_maximization_step(float *d_A, float *d_B,
                                       const float *d_A_numer, const float *d_A_denom,
                                       const float *d_O_numer, const float *d_O_denom,
                                       int N, int M)
{
    for (int flat_idx = blockIdx.x * blockDim.x + threadIdx.x; flat_idx < N * M; flat_idx += gridDim.x * blockDim.x) {
        int i = flat_idx / M;
        int k = flat_idx % M;

        if (k < N && d_A_denom[i] > 1e-9f) {
            d_A[i * N + k] = d_A_numer[i * N + k] / d_A_denom[i];
        }

        if (d_O_denom[i] > 1e-9f) {
            d_B[i * M + k] = d_O_numer[i * M + k] / d_O_denom[i];
        }
    }
}

// Initialize probabilities at t=0 for Forward and Viterbi algorithms
__global__ void kernel_init_t0(float *d_out_probs, const float *d_pi, const float *d_B, int obs0, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_out_probs[i] = d_pi[i] * d_B[i * M + obs0];
    }
}

// Forward step for probability calculation
__global__ void kernel_forward_step(float *d_alpha_t, const float *d_alpha_t_minus_1,
                                  const float *d_A, const float *d_B, int obs_t, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += d_alpha_t_minus_1[j] * d_A[j * N + i];
        }
        d_alpha_t[i] = sum * d_B[i * M + obs_t];
    }
}

// Viterbi step for finding most likely state sequence
__global__ void kernel_viterbi_step(float *d_v_t, int *d_path_t,
                                  const float *d_v_t_minus_1,
                                  const float *d_A, const float *d_B, int obs_t, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float max_prob = -1.0f;
        int best_prev_state = -1;
        for (int j = 0; j < N; ++j) {
            float prob = d_v_t_minus_1[j] * d_A[j * N + i];
            if (prob > max_prob) {
                max_prob = prob;
                best_prev_state = j;
            }
        }
        d_v_t[i] = max_prob * d_B[i * M + obs_t];
        d_path_t[i] = best_prev_state;
    }
}

// Initialize beta values for backward algorithm
__global__ void kernel_init_beta(float *d_beta_t_minus_1, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_beta_t_minus_1[i] = 1.0f;
    }
}

// Backward step for probability calculation
__global__ void kernel_backward_step(float *d_beta_t, const float *d_beta_t_plus_1,
                                   const float *d_A, const float *d_B, int obs_t1, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += d_A[i * N + j] * d_B[j * M + obs_t1] * d_beta_t_plus_1[j];
        }
        d_beta_t[i] = sum;
    }
}

// Scale a vector by a scalar value
__global__ void kernel_scale_vector(float *d_vector, float scalar, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_vector[i] *= scalar;
    }
}

// Class implementation for HMM operations on GPU
HMM_GPU::HMM_GPU(int num_states, int num_obs_symbols, int max_seq_len)
    : N(num_states), M(num_obs_symbols), max_T(max_seq_len)
{
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pi, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_obs, max_T * sizeof(int)));
    
    const int time_slots = (max_T + 1);
    CHECK_CUDA_ERROR(cudaMalloc(&d_alpha, time_slots * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta, time_slots * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gamma, time_slots * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_viterbi_probs, max_T * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_viterbi_paths, max_T * N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scaling_factors, max_T * sizeof(float)));
}

HMM_GPU::~HMM_GPU()
{
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_pi);
    cudaFree(d_obs);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_gamma);
    cudaFree(d_viterbi_probs);
    cudaFree(d_viterbi_paths);
    cudaFree(d_scaling_factors);
}

void HMM_GPU::run_forward_pass_internal(int T, const int *h_obs_for_indexing)
{
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    kernel_init_t0<<<blocks, threads_per_block>>>(d_alpha, d_pi, d_B, h_obs_for_indexing[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = 1; t < T; ++t) {
        kernel_forward_step<<<blocks, threads_per_block>>>(
            &d_alpha[t * N], &d_alpha[(t - 1) * N], d_A, d_B, h_obs_for_indexing[t], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

void HMM_GPU::run_backward_pass_internal(int T, const int *h_obs_for_indexing)
{
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    kernel_init_beta<<<blocks, threads_per_block>>>(&d_beta[(T - 1) * N], N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = T - 2; t >= 0; --t) {
        kernel_backward_step<<<blocks, threads_per_block>>>(
            &d_beta[t * N], &d_beta[(t + 1) * N], d_A, d_B, h_obs_for_indexing[t + 1], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

float HMM_GPU::forward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    if (T > max_T) {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    run_forward_pass_internal(T, h_obs);

    std::vector<float> h_alpha_final(N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_alpha_final.data(), &d_alpha[(T - 1) * N], N * sizeof(float), cudaMemcpyDeviceToHost));

    return std::accumulate(h_alpha_final.begin(), h_alpha_final.end(), 0.0f);
}

float HMM_GPU::backward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    if (T > max_T) {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    run_backward_pass_internal(T, h_obs);

    std::vector<float> h_beta0(N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_beta0.data(), d_beta, N * sizeof(float), cudaMemcpyDeviceToHost));

    float total_prob = 0.0f;
    for (int i = 0; i < N; ++i) {
        total_prob += h_pi[i] * h_B[i * M + h_obs[0]] * h_beta0[i];
    }
    return total_prob;
}

std::string HMM_GPU::viterbi(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    if (T > max_T) {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    kernel_init_t0<<<blocks, threads_per_block>>>(d_viterbi_probs, d_pi, d_B, h_obs[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = 1; t < T; ++t) {
        kernel_viterbi_step<<<blocks, threads_per_block>>>(
            &d_viterbi_probs[t * N], &d_viterbi_paths[t * N], &d_viterbi_probs[(t - 1) * N],
            d_A, d_B, h_obs[t], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    std::vector<int> h_path_matrix(T * N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_path_matrix.data(), d_viterbi_paths, T * N * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float> h_final_probs(N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_final_probs.data(), &d_viterbi_probs[(T - 1) * N], N * sizeof(float), cudaMemcpyDeviceToHost));

    int last_state = 0;
    float max_prob = -1.0f;
    for (int i = 0; i < N; ++i) {
        if (h_final_probs[i] > max_prob) {
            max_prob = h_final_probs[i];
            last_state = i;
        }
    }

    std::vector<int> optimal_path(T);
    optimal_path[T - 1] = last_state;
    for (int t = T - 2; t >= 0; --t) {
        optimal_path[t] = h_path_matrix[(t + 1) * N + optimal_path[t + 1]];
    }

    std::ostringstream ss;
    for (int state : optimal_path)
        ss << state;
    return ss.str();
}

void HMM_GPU::run_forward_pass_scaled(int T, const int *h_obs)
{
    kernel_init_alpha_scaled<<<1, N, N * sizeof(float)>>>(&d_alpha[N], &d_scaling_factors[0], d_pi, d_B, h_obs[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = 2; t <= T; ++t) {
        kernel_forward_step_scaled<<<1, N, N * sizeof(float)>>>(
            &d_alpha[t * N], &d_scaling_factors[t - 1], &d_alpha[(t - 1) * N],
            d_A, d_B, h_obs[t - 1], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // kernel_normalize<<<1, threads_per_block>>>(&d_alpha[t * N], N);
        // CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

void HMM_GPU::run_backward_pass_scaled(int T, const int *h_obs)
{
    std::vector<float> h_scaling(T);
    CHECK_CUDA_ERROR(cudaMemcpy(h_scaling.data(), d_scaling_factors, T * sizeof(float), cudaMemcpyDeviceToHost));

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    kernel_init_beta<<<blocks, threads_per_block>>>(&d_beta[T * N], N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = T - 1; t >= 1; --t) {
        kernel_backward_step_scaled<<<blocks, threads_per_block>>>(
            &d_beta[t * N],
            &d_beta[(t + 1) * N],
            d_A,
            d_B,
            h_obs[t],
            h_scaling[t],
            N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // kernel_normalize<<<blocks, threads_per_block>>>(d_beta, N);
        // CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

void HMM_GPU::baum_welch(const int *h_obs, float *h_A, float *h_B, float *h_pi, int T, int max_iters, float tolerance)
{
    if (T > max_T) {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }

    float *d_A_numer, *d_A_denom, *d_O_numer, *d_O_denom;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A_numer, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_A_denom, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O_numer, N * M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_O_denom, N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<int> h_obs_vec(h_obs, h_obs + T);

    for (int iter = 0; iter < max_iters; ++iter) {
        CHECK_CUDA_ERROR(cudaMemset(d_A_numer, 0, N * N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_A_denom, 0, N * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_O_numer, 0, N * M * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemset(d_O_denom, 0, N * sizeof(float)));

        run_forward_pass_scaled(T, h_obs_vec.data());
        run_backward_pass_scaled(T, h_obs_vec.data());

        kernel_expectation_step<<<T, 256, 256 * sizeof(float)>>>(
            d_A_numer, d_A_denom, d_O_numer, d_O_denom,
            d_alpha, d_beta, d_A, d_B, d_obs, N, M, T);
        CHECK_CUDA_ERROR(cudaGetLastError());

        const int m_threads = 256;
        const int m_blocks = (N * M + m_threads - 1) / m_threads;
        kernel_maximization_step<<<m_blocks, m_threads>>>(d_A, d_B, d_A_numer, d_A_denom, d_O_numer, d_O_denom, N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    CHECK_CUDA_ERROR(cudaMemcpy(h_A, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A_numer);
    cudaFree(d_A_denom);
    cudaFree(d_O_numer);
    cudaFree(d_O_denom);
}