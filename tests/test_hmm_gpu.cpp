#include "hmm_gpu.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

// Include the provided CPU implementation for comparison
#include "hmm.hpp" // Assumes your CPU code is in hmm.hpp/cpp
#include "data_loader.hpp"

// Helper to generate random HMM data for testing
void generate_random_hmm_data(std::vector<float>& A, std::vector<float>& B, std::vector<float>& pi, std::vector<int>& obs, int N, int M, int T) {
    A.resize(N * N);
    B.resize(N * M);
    pi.resize(N);
    obs.resize(T);

    std::mt19937 gen(1337);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; ++j) { A[i * N + j] = dis(gen); row_sum += A[i * N + j]; }
        for (int j = 0; j < N; ++j) A[i * N + j] /= row_sum;
    }
    for (int i = 0; i < N; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < M; ++j) { B[i * M + j] = dis(gen); row_sum += B[i * M + j]; }
        for (int j = 0; j < M; ++j) B[i * M + j] /= row_sum;
    }
    float pi_sum = 0.0f;
    for (int i = 0; i < N; ++i) { pi[i] = dis(gen); pi_sum += pi[i]; }
    for (int i = 0; i < N; ++i) pi[i] /= pi_sum;

    std::uniform_int_distribution<> obs_dis(0, M - 1);
    for (int t = 0; t < T; ++t) obs[t] = obs_dis(gen);
}

void run_benchmarks() {
    std::cout << "\n--- HMM GPU vs CPU Benchmark ---" << std::endl;
    std::cout << std::string(85, '=') << std::endl;

    std::vector<int> N_vals = {16, 32, 64, 128};
    std::vector<int> T_vals = {1024, 2048, 4096};
    const int M = 32;
    const int max_iters = 5;

    std::cout << std::left << std::setw(8) << "N" << std::setw(8) << "T"
              << std::setw(18) << "Viterbi Speedup"
              << std::setw(18) << "Forward Speedup"
              << std::setw(25) << "Baum-Welch Speedup" << std::endl;
    std::cout << std::string(85, '-') << std::endl;

    for (int N : N_vals) {
        for (int T : T_vals) {
            std::vector<float> h_A, h_B, h_pi;
            std::vector<int> h_obs;
            generate_random_hmm_data(h_A, h_B, h_pi, h_obs, N, M, T);

            IHMM hmm_cpu(N, M);
            HMM_GPU hmm_gpu(N, M, T);
            
            double viterbi_speedup = 0.0, forward_speedup = 0.0, bw_speedup = 0.0;

            // Viterbi Benchmark
            {
                auto start_cpu = std::chrono::high_resolution_clock::now();
                hmm_cpu.viterbi(h_obs.data(), nullptr, h_pi.data(), h_A.data(), h_B.data(), T, N, M);
                auto end_cpu = std::chrono::high_resolution_clock::now();
                auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

                auto start_gpu = std::chrono::high_resolution_clock::now();
                hmm_gpu.viterbi(h_obs.data(), h_A.data(), h_B.data(), h_pi.data(), T);
                cudaDeviceSynchronize();
                auto end_gpu = std::chrono::high_resolution_clock::now();
                auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();
                if (gpu_duration > 0) viterbi_speedup = static_cast<double>(cpu_duration) / gpu_duration;
            }

            // Baum-Welch Benchmark
            {
                auto h_A_cpu = h_A, h_B_cpu = h_B, h_pi_cpu = h_pi;
                auto h_A_gpu = h_A, h_B_gpu = h_B, h_pi_gpu = h_pi;
                
                auto start_cpu = std::chrono::high_resolution_clock::now();
                hmm_cpu.baum_welch(h_obs.data(), nullptr, h_pi_cpu.data(), h_A_cpu.data(), h_B_cpu.data(), T, N, M, max_iters);
                auto end_cpu = std::chrono::high_resolution_clock::now();
                auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

                auto start_gpu = std::chrono::high_resolution_clock::now();
                hmm_gpu.baum_welch(h_obs.data(), h_A_gpu.data(), h_B_gpu.data(), h_pi_gpu.data(), T, max_iters, 1e-4);
                cudaDeviceSynchronize();
                auto end_gpu = std::chrono::high_resolution_clock::now();
                auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();
                if (gpu_duration > 0) bw_speedup = static_cast<double>(cpu_duration) / gpu_duration;
            }

            std::cout << std::left << std::fixed << std::setprecision(2)
                      << std::setw(8) << N << std::setw(8) << T
                      << std::setw(18) << std::to_string(viterbi_speedup) + "x"
                      << std::setw(18) << "N/A" // Forward is part of BW
                      << std::setw(25) << std::to_string(bw_speedup) + "x" << std::endl;
        }
    }
}

int main() {
    std::cout << "HMM GPU Implementation Tests & Benchmarks" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // You can add correctness tests here by comparing results from CPU and GPU
    // For example, load a sequence file and check if viterbi paths match.
    
    run_benchmarks();
    
    return 0;
}