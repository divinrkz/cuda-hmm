// File: hmm_gpu.cu
#include "hmm_gpu.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cmath>
#include <sstream>
#include <limits>
#include <numeric>

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA_ERROR(err)                                                                              \
    if (err != cudaSuccess)                                                                                \
    {                                                                                                      \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                                \
    }

// ========================================================================== //
//                STATE-PARALLEL KERNELS (NO TEMPORAL PARALLELISM)            //
// ========================================================================== //

// Kernel: Initialization for Forward and Viterbi at t=0
__global__ void kernel_init_t0(float *d_out_probs, const float *d_pi, const float *d_B, int obs0, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_out_probs[i] = d_pi[i] * d_B[i * M + obs0];
    }
}

// Kernel: Forward recursive step for one time step t.
// alpha_t(i) = [ sum_j(alpha_{t-1}(j) * A_ji) ] * B_i(O_t)
__global__ void kernel_forward_step(float *d_alpha_t, const float *d_alpha_t_minus_1,
                                    const float *d_A, const float *d_B, int obs_t, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current state
    if (i < N)
    {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
        { // previous state j
            sum += d_alpha_t_minus_1[j] * d_A[j * N + i];
        }
        d_alpha_t[i] = sum * d_B[i * M + obs_t];
    }
}

// Kernel: Viterbi recursive step for one time step t.
// v_t(i) = [ max_j(v_{t-1}(j) * A_ji) ] * B_i(O_t)
__global__ void kernel_viterbi_step(float *d_v_t, int *d_path_t,
                                    const float *d_v_t_minus_1,
                                    const float *d_A, const float *d_B, int obs_t, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current state i
    if (i < N)
    {
        float max_prob = -1.0f;
        int best_prev_state = -1;
        for (int j = 0; j < N; ++j)
        { // previous state j
            float prob = d_v_t_minus_1[j] * d_A[j * N + i];
            if (prob > max_prob)
            {
                max_prob = prob;
                best_prev_state = j;
            }
        }
        d_v_t[i] = max_prob * d_B[i * M + obs_t];
        d_path_t[i] = best_prev_state;
    }
}

// Kernel: Backward initialization at t=T-1
__global__ void kernel_init_beta(float *d_beta_t_minus_1, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_beta_t_minus_1[i] = 1.0f;
    }
}

// Kernel: Backward recursive step for one time step t.
// beta_t(i) = sum_j(A_ij * B_j(O_{t+1}) * beta_{t+1}(j))
__global__ void kernel_backward_step(float *d_beta_t, const float *d_beta_t_plus_1,
                                     const float *d_A, const float *d_B, int obs_t1, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // current state i
    if (i < N)
    {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
        { // next state j
            sum += d_A[i * N + j] * d_B[j * M + obs_t1] * d_beta_t_plus_1[j];
        }
        d_beta_t[i] = sum;
    }
}

// Kernel: Computes gamma and xi for all t < T-1.
__global__ void kernel_compute_gamma_xi(float *d_gamma, float *d_xi, const float *d_alpha,
                                        const float *d_beta, const float *d_A, const float *d_B,
                                        const int *d_obs, int N, int M, int T)
{
    int t = blockIdx.x;
    if (t >= T - 1)
        return;

    // Denominator calculation (can be a performance bottleneck, but safe)
    float denom = 0.0f;
    int obs_t1 = d_obs[t + 1];
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            denom += d_alpha[t * N + i] * d_A[i * N + j] * d_B[j * M + obs_t1] * d_beta[(t + 1) * N + j];
        }
    }

    if (denom < 1e-35f)
    { // Avoid division by zero
        for (int i = threadIdx.x; i < N; i += blockDim.x)
        {
            d_gamma[t * N + i] = 1.0f / N; // Fallback to uniform
            for (int j = 0; j < N; ++j)
                d_xi[(t * N + i) * N + j] = 1.0f / (N * N);
        }
        return;
    }

    // Main calculation loop for each thread
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        float gamma_val_i = 0.0f;
        for (int j = 0; j < N; ++j)
        {
            float xi_val = (d_alpha[t * N + i] * d_A[i * N + j] * d_B[j * M + obs_t1] * d_beta[(t + 1) * N + j]) / denom;
            d_xi[(t * N + i) * N + j] = xi_val;
            gamma_val_i += xi_val;
        }
        d_gamma[t * N + i] = gamma_val_i;
    }
}

// Kernel: Computes the last gamma column gamma(T-1) separately.
__global__ void kernel_compute_last_gamma(float *d_gamma, const float *d_alpha, int N, int T)
{
    float denom = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        denom += d_alpha[(T - 1) * N + i];
    }

    if (denom < 1e-35f)
    { // Avoid division by zero
        for (int i = threadIdx.x; i < N; i += blockDim.x)
            d_gamma[(T - 1) * N + i] = 1.0f / N;
        return;
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_gamma[(T - 1) * N + i] = d_alpha[(T - 1) * N + i] / denom;
    }
}

// Kernel: M-Step - updates pi, A, and B.
__global__ void kernel_m_step(float *d_pi, float *d_A, float *d_B, const float *d_gamma,
                              const float *d_xi, const int *d_obs, int N, int M, int T)
{
    int i = blockIdx.x;
    if (i >= N)
        return;

    // Update pi
    d_pi[i] = d_gamma[i];

    // Update A
    float gamma_sum_A = 0.0f;
    for (int t = 0; t < T - 1; ++t)
    {
        gamma_sum_A += d_gamma[t * N + i];
    }
    if (gamma_sum_A > 1e-9)
    {
        for (int j = threadIdx.x; j < N; j += blockDim.x)
        {
            float xi_sum = 0.0f;
            for (int t = 0; t < T - 1; ++t)
            {
                xi_sum += d_xi[(t * N + i) * N + j];
            }
            d_A[i * N + j] = xi_sum / gamma_sum_A;
        }
    }

    // Update B
    float gamma_sum_B = gamma_sum_A + d_gamma[(T - 1) * N + i];
    if (gamma_sum_B > 1e-9)
    {
        for (int k = threadIdx.x; k < M; k += blockDim.x)
        {
            float numer = 0.0f;
            for (int t = 0; t < T; ++t)
            {
                if (d_obs[t] == k)
                {
                    numer += d_gamma[t * N + i];
                }
            }
            d_B[i * M + k] = numer / gamma_sum_B;
        }
    }
}

// ========================================================================== //
//                    CLASS MEMBER IMPLEMENTATIONS
// ========================================================================== //

HMM_GPU::HMM_GPU(int num_states, int num_obs_symbols, int max_seq_len)
    : N(num_states), M(num_obs_symbols), max_T(max_seq_len)
{
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * M * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_pi, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_obs, max_T * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_alpha, max_T * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta, max_T * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_gamma, max_T * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_xi, (max_T - 1) * N * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_viterbi_probs, max_T * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_viterbi_paths, max_T * N * sizeof(int)));
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
    cudaFree(d_xi);
    cudaFree(d_viterbi_probs);
    cudaFree(d_viterbi_paths);
}

// --- PUBLIC API FUNCTIONS ---

float HMM_GPU::forward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    if (T > max_T)
    {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }
    // 1. Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // 2. Initialization t=0
    kernel_init_t0<<<blocks, threads_per_block>>>(d_alpha, d_pi, d_B, h_obs[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 3. Induction loop on host
    for (int t = 1; t < T; ++t)
    {
        kernel_forward_step<<<blocks, threads_per_block>>>(
            &d_alpha[t * N], &d_alpha[(t - 1) * N], d_A, d_B, h_obs[t], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // 4. Final probability calculation
    std::vector<float> h_alpha_final(N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_alpha_final.data(), &d_alpha[(T - 1) * N], N * sizeof(float), cudaMemcpyDeviceToHost));

    float total_prob = std::accumulate(h_alpha_final.begin(), h_alpha_final.end(), 0.0f);
    return total_prob;
}

float HMM_GPU::backward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    if (T > max_T)
    {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }
    // 1. Copy data
    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // 2. Initialization t=T-1
    kernel_init_beta<<<blocks, threads_per_block>>>(&d_beta[(T - 1) * N], N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 3. Induction loop
    for (int t = T - 2; t >= 0; --t)
    {
        kernel_backward_step<<<blocks, threads_per_block>>>(
            &d_beta[t * N], &d_beta[(t + 1) * N], d_A, d_B, h_obs[t + 1], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // 4. Final probability calculation
    std::vector<float> h_beta0(N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_beta0.data(), d_beta, N * sizeof(float), cudaMemcpyDeviceToHost));

    float total_prob = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        total_prob += h_pi[i] * h_B[i * M + h_obs[0]] * h_beta0[i];
    }
    return total_prob;
}

std::string HMM_GPU::viterbi(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    if (T > max_T)
    {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }
    // 1. Copy data
    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // 2. Initialization t=0
    kernel_init_t0<<<blocks, threads_per_block>>>(d_viterbi_probs, d_pi, d_B, h_obs[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 3. Induction loop
    for (int t = 1; t < T; ++t)
    {
        kernel_viterbi_step<<<blocks, threads_per_block>>>(
            &d_viterbi_probs[t * N], &d_viterbi_paths[t * N], &d_viterbi_probs[(t - 1) * N],
            d_A, d_B, h_obs[t], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    // 4. Backtracking on Host
    std::vector<int> h_path_matrix(T * N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_path_matrix.data(), d_viterbi_paths, T * N * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<float> h_final_probs(N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_final_probs.data(), &d_viterbi_probs[(T - 1) * N], N * sizeof(float), cudaMemcpyDeviceToHost));

    int last_state = 0;
    float max_prob = -1.0f;
    for (int i = 0; i < N; ++i)
    {
        if (h_final_probs[i] > max_prob)
        {
            max_prob = h_final_probs[i];
            last_state = i;
        }
    }

    std::vector<int> optimal_path(T);
    optimal_path[T - 1] = last_state;
    for (int t = T - 2; t >= 0; --t)
    {
        optimal_path[t] = h_path_matrix[(t + 1) * N + optimal_path[t + 1]];
    }

    std::ostringstream ss;
    for (int state : optimal_path)
        ss << state;
    return ss.str();
}

// --- PRIVATE HELPERS for BAUM-WELCH ---
// (Assumes d_A, d_B, d_pi, d_obs are already on the device)

void HMM_GPU::_forward_internal(const int *h_obs, int T)
{
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Initialization t=0 (using d_pi)
    kernel_init_t0<<<blocks, threads_per_block>>>(d_alpha, d_pi, d_B, h_obs[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Induction loop on host
    for (int t = 1; t < T; ++t)
    {
        kernel_forward_step<<<blocks, threads_per_block>>>(
            &d_alpha[t * N], &d_alpha[(t - 1) * N], d_A, d_B, h_obs[t], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

void HMM_GPU::_backward_internal(const int *h_obs, int T)
{
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    // Initialization t=T-1
    kernel_init_beta<<<blocks, threads_per_block>>>(&d_beta[(T - 1) * N], N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Induction loop
    for (int t = T - 2; t >= 0; --t)
    {
        kernel_backward_step<<<blocks, threads_per_block>>>(
            &d_beta[t * N], &d_beta[(t + 1) * N], d_A, d_B, h_obs[t + 1], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

void HMM_GPU::baum_welch(const int *h_obs, float *h_A, float *h_B, float *h_pi, int T, int max_iters, float tolerance)
{
    if (T > max_T)
    {
        throw std::runtime_error("Sequence length T exceeds max_T.");
    }
    // 1. Copy initial data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    float old_prob = -1.0f;
    const int threads_per_block = 256;

    for (int iter = 0; iter < max_iters; ++iter)
    {
        // --- E-Step ---
        // Run forward and backward passes using internal methods that operate
        // on device memory directly, avoiding redundant H->D copies.
        this->_forward_internal(h_obs, T);
        this->_backward_internal(h_obs, T);

        // Check for convergence
        std::vector<float> h_alpha_final(N);
        CHECK_CUDA_ERROR(cudaMemcpy(h_alpha_final.data(), &d_alpha[(T - 1) * N], N * sizeof(float), cudaMemcpyDeviceToHost));
        float new_prob = std::accumulate(h_alpha_final.begin(), h_alpha_final.end(), 0.0f);

        if (fabsf(new_prob - old_prob) < tolerance)
        {
            break;
        }
        old_prob = new_prob;

        // Compute gamma and xi
        const int gamma_blocks = (N + threads_per_block - 1) / threads_per_block;
        kernel_compute_gamma_xi<<<T - 1, threads_per_block>>>(d_gamma, d_xi, d_alpha, d_beta, d_A, d_B, d_obs, N, M, T);
        CHECK_CUDA_ERROR(cudaGetLastError());
        kernel_compute_last_gamma<<<gamma_blocks, threads_per_block>>>(d_gamma, d_alpha, N, T);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // --- M-Step ---
        kernel_m_step<<<N, threads_per_block>>>(d_pi, d_A, d_B, d_gamma, d_xi, d_obs, N, M, T);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Sync device to ensure M-step kernel is complete before next iteration's E-step
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // --- Final Step ---
    // Copy the final trained parameters from device back to the host pointers.
    CHECK_CUDA_ERROR(cudaMemcpy(h_A, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pi, d_pi, N * sizeof(float), cudaMemcpyDeviceToHost));
}