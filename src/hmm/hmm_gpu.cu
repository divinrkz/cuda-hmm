// File: hmm_gpu.cu
#include "hmm_gpu.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <vector>
#include <numeric>
#include <cmath>
#include <sstream>

// --- CUDA/CUBLAS Error Checking ---
#define CHECK_CUDA_ERROR(err)                                                                              \
    if (err != cudaSuccess)                                                                                \
    {                                                                                                      \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                                \
    }

const char *cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "Unknown cuBLAS error";
}

#define CHECK_CUBLAS_ERROR(err)                                                                                \
    if (err != CUBLAS_STATUS_SUCCESS)                                                                          \
    {                                                                                                          \
        fprintf(stderr, "cuBLAS error in %s at line %d: %s\n", __FILE__, __LINE__, cublasGetErrorString(err)); \
        exit(EXIT_FAILURE);                                                                                    \
    }

// ========================================================================== //
//                                KERNELS                                     //
// ========================================================================== //

__global__ void kernel_init_alpha_unscaled(float *d_alpha, const float *d_pi, const float *d_B, int obs0, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_alpha[i] = d_pi[i] * d_B[i * M + obs0];
    }
}

__global__ void kernel_update_alpha_emission(float *d_alpha_t, const float *d_B, int obs_t, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_alpha_t[i] *= d_B[i * M + obs_t];
    }
}

__global__ void kernel_init_beta_unscaled(float *d_beta_T, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_beta_T[i] = 1.0f;
    }
}

__global__ void kernel_update_beta_emission(float *d_beta_t1, const float *d_B, int obs_t1, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_beta_t1[i] *= d_B[i * M + obs_t1];
    }
}

__global__ void kernel_compute_gamma_xi(float *d_gamma, float *d_xi,
                                        const float *d_alpha, const float *d_beta, const float *d_A,
                                        const float *d_B, const int *d_obs, int N, int M, int T)
{
    int t = blockIdx.x;
    if (t >= T - 1)
        return;

    // This kernel now uses a 1D block. We need to calculate the denominator first.
    // This is less efficient than the previous shared memory approach but safer across all problem sizes.
    // A full high-performance version would use a block-level reduction here.
    float denom = 0.0f;
    for (int i_denom = 0; i_denom < N; ++i_denom)
    {
        for (int j_denom = 0; j_denom < N; ++j_denom)
        {
            denom += d_alpha[t * N + i_denom] * d_A[i_denom * N + j_denom] * d_B[j_denom * M + d_obs[t + 1]] * d_beta[(t + 1) * N + j_denom];
        }
    }
    if (denom < 1e-35f)
        return;

    // Grid-stride loop for i
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        float gamma_val = 0.0f;
        for (int j = 0; j < N; ++j)
        {
            float xi_val = (d_alpha[t * N + i] * d_A[i * N + j] * d_B[j * M + d_obs[t + 1]] * d_beta[(t + 1) * N + j]) / denom;
            d_xi[(t * N + i) * N + j] = xi_val;
            gamma_val += xi_val;
        }
        d_gamma[t * N + i] = gamma_val;
    }
}

__global__ void kernel_compute_last_gamma(float *d_gamma, const float *d_alpha, int N, int T)
{
    // In an unscaled implementation, gamma at T-1 is just the normalized alpha at T-1.
    float denom = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        denom += d_alpha[(T - 1) * N + i];
    }
    if (denom == 0.0f)
        return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_gamma[(T - 1) * N + i] = d_alpha[(T - 1) * N + i] / denom;
    }
}

__global__ void kernel_viterbi_init(float *d_v, const float *d_pi, const float *d_B, int obs0, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        d_v[i] = d_pi[i] * d_B[i * M + obs0];
    }
}

__global__ void kernel_viterbi_step(float *d_v_curr, int *d_path_curr, const float *d_v_prev,
                                    const float *d_A, const float *d_B, int obs_t, int N, int M)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N)
    {
        float max_prob = 0.0f;
        int best_path_idx = -1;

        for (int i = 0; i < N; ++i)
        {
            float prob = d_v_prev[i] * d_A[i * N + j];
            if (prob > max_prob)
            {
                max_prob = prob;
                best_path_idx = i;
            }
        }

        d_v_curr[j] = max_prob * d_B[j * M + obs_t];
        d_path_curr[j] = best_path_idx;
    }
}

__global__ void kernel_m_step(float *d_pi, float *d_A, float *d_B, const float *d_gamma, const float *d_xi, const int *d_obs, int N, int M, int T)
{
    int i = blockIdx.x;
    if (i >= N)
        return;

    d_pi[i] = d_gamma[i];

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
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_vec, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_viterbi_probs, max_T * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_viterbi_paths, max_T * N * sizeof(int)));
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_handle));
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
    cudaFree(d_temp_vec);
    cudaFree(d_viterbi_probs);
    cudaFree(d_viterbi_paths);
    CHECK_CUBLAS_ERROR(cublasDestroy(cublas_handle));
}

void HMM_GPU::run_forward_pass(int T)
{
    const float alpha_blas = 1.0f;
    const float beta_blas = 0.0f;
    std::vector<int> h_obs(T);
    CHECK_CUDA_ERROR(cudaMemcpy(h_obs.data(), d_obs, T * sizeof(int), cudaMemcpyDeviceToHost));

    kernel_init_alpha_unscaled<<<(N + 255) / 256, 256>>>(d_alpha, d_pi, d_B, h_obs[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = 1; t < T; ++t)
    {
        float *d_alpha_prev = &d_alpha[(t - 1) * N];
        float *d_alpha_curr = &d_alpha[t * N];
        CHECK_CUBLAS_ERROR(cublasSgemv(cublas_handle, CUBLAS_OP_T, N, N, &alpha_blas, d_A, N, d_alpha_prev, 1, &beta_blas, d_alpha_curr, 1));
        kernel_update_alpha_emission<<<(N + 255) / 256, 256>>>(d_alpha_curr, d_B, h_obs[t], N, M);
    }
}

void HMM_GPU::run_backward_pass(int T)
{
    const float alpha_blas = 1.0f;
    const float beta_blas = 0.0f;
    std::vector<int> h_obs(T);
    CHECK_CUDA_ERROR(cudaMemcpy(h_obs.data(), d_obs, T * sizeof(int), cudaMemcpyDeviceToHost));

    kernel_init_beta_unscaled<<<(N + 255) / 256, 256>>>(&d_beta[(T - 1) * N], N);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = T - 2; t >= 0; --t)
    {
        float *d_beta_prev = &d_beta[(t + 1) * N];
        float *d_beta_curr = &d_beta[t * N];
        CHECK_CUDA_ERROR(cudaMemcpy(d_temp_vec, d_beta_prev, N * sizeof(float), cudaMemcpyDeviceToDevice));
        kernel_update_beta_emission<<<(N + 255) / 256, 256>>>(d_temp_vec, d_B, h_obs[t + 1], N, M);
        CHECK_CUBLAS_ERROR(cublasSgemv(cublas_handle, CUBLAS_OP_N, N, N, &alpha_blas, d_A, N, d_temp_vec, 1, &beta_blas, d_beta_curr, 1));
    }
}

float HMM_GPU::forward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    run_forward_pass(T);

    float total_prob = 0.0f;
    CHECK_CUBLAS_ERROR(cublasSasum(cublas_handle, N, &d_alpha[(T - 1) * N], 1, &total_prob));
    return total_prob;
}

float HMM_GPU::backward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    run_backward_pass(T);

    float total_prob = 0.0f;
    std::vector<float> h_beta0(N);

    CHECK_CUDA_ERROR(cudaMemcpy(h_beta0.data(), &d_beta[0], N * sizeof(float), cudaMemcpyDeviceToHost));

    // Use the h_obs, h_pi, and h_B pointers passed directly into the function.
    // No need for intermediate vectors or extra copies from device.
    int obs0 = h_obs[0];
    for (int i = 0; i < N; ++i)
    {
        total_prob += h_pi[i] * h_B[i * M + obs0] * h_beta0[i];
    }
    return total_prob;
}

std::string HMM_GPU::viterbi(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T)
{
    if (T > max_T)
    {
        std::cerr << "Error: Sequence length T=" << T << " exceeds max_T=" << max_T << " allocated for HMM_GPU object." << std::endl;
        return "";
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    kernel_viterbi_init<<<(N + 255) / 256, 256>>>(d_viterbi_probs, d_pi, d_B, h_obs[0], N, M);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int t = 1; t < T; ++t)
    {
        float *d_v_prev = &d_viterbi_probs[(t - 1) * N];
        float *d_v_curr = &d_viterbi_probs[t * N];
        int *d_path_curr = &d_viterbi_paths[t * N];

        kernel_viterbi_step<<<(N + 255) / 256, 256>>>(d_v_curr, d_path_curr, d_v_prev, d_A, d_B, h_obs[t], N, M);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

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
    for (int i = 0; i < T; ++i)
    {
        ss << optimal_path[i];
    }

    return ss.str();
}

void HMM_GPU::baum_welch(const int *h_obs, float *h_A, float *h_B, float *h_pi, int T, int max_iters, float tolerance)
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_obs, h_obs, T * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pi, h_pi, N * sizeof(float), cudaMemcpyHostToDevice));

    float old_prob = -1.0;

    for (int iter = 0; iter < max_iters; ++iter)
    {
        run_forward_pass(T);
        run_backward_pass(T);

        float new_prob = 0.0f;
        CHECK_CUBLAS_ERROR(cublasSasum(cublas_handle, N, &d_alpha[(T - 1) * N], 1, &new_prob));

        if (fabsf(new_prob - old_prob) < tolerance)
        {
            break;
        }
        old_prob = new_prob;

        // E-Step: Compute gamma and xi
        const int threads = 256;
        kernel_compute_gamma_xi<<<T - 1, threads>>>(d_gamma, d_xi, d_alpha, d_beta, d_A, d_B, d_obs, N, M, T);
        CHECK_CUDA_ERROR(cudaGetLastError());
        kernel_compute_last_gamma<<<(N + 255) / 256, 256>>>(d_gamma, d_alpha, N, T);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // M-Step: Update pi, A, B
        kernel_m_step<<<N, 256>>>(d_pi, d_A, d_B, d_gamma, d_xi, d_obs, N, M, T);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    CHECK_CUDA_ERROR(cudaMemcpy(h_A, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_B, h_B, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_pi, d_pi, N * sizeof(float), cudaMemcpyDeviceToHost));
}