#include "hmm.hpp"
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>

// Helper function to format duration
std::string format_duration(std::chrono::microseconds duration)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    if (duration.count() < 1000)
    {
        ss << duration.count() << " µs";
    }
    else if (duration.count() < 1000000)
    {
        ss << (duration.count() / 1000.0) << " ms";
    }
    else
    {
        ss << (duration.count() / 1000000.0) << " s";
    }
    return ss.str();
}

float **IHMM::forward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M)
{
    auto start = std::chrono::high_resolution_clock::now();

    // CUDA PARALLELIZATION STRATEGY FOR FORWARD ALGORITHM:
    // 1. TRELLIS PARALLELIZATION: Create a T×N trellis (matrix) where each cell α(t,i) represents
    //    the probability of being in state i at time t. Each time step depends on the previous one,
    //    but within each time step, all states can be computed in parallel.
    //
    // 2. THREAD MAPPING: Launch N threads per time step, where each thread computes one state:
    //    - threadIdx.x = state_index (0 to N-1)
    //    - blockIdx.x = time_step (2 to T, since step 1 is initialization)
    //
    // 3. PARALLEL REDUCTION: The inner loop (summing over previous states) can use
    //    parallel reduction to compute the sum more efficiently than sequential addition

    // Create and initialize alphas (T+1 x N matrix)
    float **alphas = new float *[T + 1];
    for (int t = 0; t <= T; t++)
    {
        alphas[t] = new float[N](); // initialize with zeros
    }

    // Initialize first time step
    // CUDA: This initialization can be parallelized with N threads
    for (int i = 0; i < N; i++)
    {
        alphas[1][i] = start_p[i] * emit_p[i * M + static_cast<int>(obs[0])];
    }

    // Recursive step
    // CUDA: Each time step t requires synchronization, but within each time step,
    // all N states can be computed in parallel
    for (int t = 2; t <= T; t++)
    {
        // CUDA: Each thread handles one curr_state
        for (int curr_state = 0; curr_state < N; curr_state++)
        {
            float prob = 0.0f;
            // CUDA: This inner loop is perfect for parallel reduction
            // Instead of sequential sum, use parallel reduction with shared memory
            for (int prev_state = 0; prev_state < N; prev_state++)
            {
                prob += emit_p[curr_state * M + static_cast<int>(obs[t - 1])] *
                        (alphas[t - 1][prev_state] * trans_p[prev_state * N + curr_state]);
            }
            alphas[t][curr_state] = prob;
        }
        // CUDA: __syncthreads() or grid synchronization needed here before next time step
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Forward algorithm execution time: " << format_duration(duration) << std::endl;

    return alphas;
}

float **IHMM::backward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M)
{
    auto start = std::chrono::high_resolution_clock::now();

    // CUDA PARALLELIZATION STRATEGY FOR BACKWARD ALGORITHM:
    // 1. REVERSE TRELLIS: Similar to forward but processes time steps in reverse (T to 0)
    // 2. SAME THREAD MAPPING: N threads per time step, each computing one state
    // 3. MEMORY COALESCING: Ensure memory access patterns are coalesced for better performance
    // 4. SHARED MEMORY: Use shared memory to store betas values for current time step

    // Create and initialize betas (T+1 x N matrix)
    float **betas = new float *[T + 1];
    for (int t = 0; t <= T; t++)
    {
        betas[t] = new float[N](); // initialize with zeros
    }

    // Initialize last time step
    // CUDA: Parallel initialization with N threads
    for (int i = 0; i < N; i++)
    {
        betas[T][i] = 1.0f;
    }

    // Recursive step (going backwards in time)
    // CUDA: Process time steps sequentially, but parallelize within each time step
    for (int t = T - 1; t >= 0; t--)
    {
        // CUDA KERNEL: Each thread computes one curr_state
        for (int curr_state = 0; curr_state < N; curr_state++)
        {
            float prob = 0.0f;
            // CUDA: Parallel reduction opportunity here
            for (int next_state = 0; next_state < N; next_state++)
            {
                if (t == 0)
                {
                    // Special case for t=0: use start probabilities
                    prob += betas[t + 1][next_state] * start_p[next_state] * emit_p[next_state * M + static_cast<int>(obs[t])];
                }
                else
                {
                    // Regular case: use transition probabilities
                    prob += betas[t + 1][next_state] * trans_p[curr_state * N + next_state] * emit_p[next_state * M + static_cast<int>(obs[t])];
                }
            }
            betas[t][curr_state] = prob;
        }
        // CUDA: Synchronization needed before processing previous time step
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Backward algorithm execution time: " << format_duration(duration) << std::endl;

    return betas;
}

void IHMM::baum_welch(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M, int N_iters)
{
    // CUDA PARALLELIZATION STRATEGY FOR BAUM-WELCH ALGORITHM:
    // 1. FORWARD-BACKWARD PARALLELIZATION: Use parallel forward and backward algorithms
    // 2. GAMMA COMPUTATION: Parallelize across T×N grid (time×states)
    // 3. XI COMPUTATION: Parallelize across T×N×N grid (time×state_i×state_j)
    // 4. PARAMETER UPDATE: Parallelize the maximization step updates
    // 5. 2D REDUCTION: Use 2D parallel reduction for computing parameter updates
    // 6. MULTIPLE SEQUENCES: If processing multiple sequences, add another dimension of parallelism

    // assumes start_p, trans_p, emit_p are randomly initialized
    auto start = std::chrono::high_resolution_clock::now();

    float **alphas = nullptr;
    float **betas = nullptr;
    float **gamma = new float *[T];
    for (int t = 0; t < T; t++)
    {
        gamma[t] = new float[N]();
    }

    float **xi = new float *[T];
    for (int t = 0; t < T; t++)
    {
        xi[t] = new float[N * N]();
    }

    for (int iter = 0; iter < N_iters; iter++)
    {
        // ===== Expectation Step =====
        // CUDA: These forward/backward calls already have internal parallelization
        // Calculate forward and backward probabilities
        alphas = forward(obs, states, start_p, trans_p, emit_p, T, N, M);
        betas = backward(obs, states, start_p, trans_p, emit_p, T, N, M);

        // Compute gamma (probability of being in state i at time t)
        // CUDA PARALLELIZATION: Launch T×N threads in a 2D grid
        // - blockIdx.x = time_step, threadIdx.x = state_index
        for (int t = 0; t < T; t++)
        {
            float denom = 0.0f;
            // CUDA: Parallel reduction to compute denominator
            for (int i = 0; i < N; i++)
            {
                denom += alphas[t][i] * betas[t][i];
            }
            // CUDA: Each thread computes gamma for one (t,i) pair
            for (int i = 0; i < N; i++)
            {
                gamma[t][i] = alphas[t][i] * betas[t][i] / denom;
            }
        }

        // Compute xi (probability of being in state i at t and state j at t+1)
        // CUDA PARALLELIZATION: Launch (T-1)×N×N threads in a 3D grid
        // - blockIdx.x = time_step, threadIdx.x = state_i, threadIdx.y = state_j
        for (int t = 0; t < T - 1; t++)
        {
            float denom = 0.0f;
            // CUDA: 2D parallel reduction to compute denominator
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    xi[t][i * N + j] = alphas[t][i] * trans_p[i * N + j] * emit_p[j * M + static_cast<int>(obs[t + 1])] * betas[t + 1][j];
                    denom += xi[t][i * N + j];
                }
            }
            // Normalize xi
            // CUDA: Parallel normalization with N×N threads
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    xi[t][i * N + j] /= denom;
                }
            }
        }

        // ===== Maximization Step =====
        // Update model parameters using computed expectations

        // Update initial state probabilities
        // CUDA: Simple parallel copy with N threads
        for (int i = 0; i < N; i++)
        {
            start_p[i] = gamma[0][i];
        }

        // Update transition probabilities
        // CUDA PARALLELIZATION: Launch N×N threads for parallel computation
        // Each thread computes one transition probability trans_p[i][j]
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float numer = 0.0f;
                float denom = 0.0f;
                // CUDA: Use parallel reduction over time dimension
                for (int t = 0; t < T - 1; t++)
                {
                    numer += xi[t][i * N + j];
                    denom += gamma[t][i];
                }
                trans_p[i * N + j] = numer / denom;
            }
        }

        // Update emission probabilities
        // CUDA PARALLELIZATION: Launch N×M threads for parallel computation
        // Each thread computes one emission probability emit_p[i][j]
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < M; j++)
            {
                float numer = 0.0f;
                float denom = 0.0f;
                // CUDA: Use parallel reduction over time dimension
                for (int t = 0; t < T; t++)
                {
                    if (static_cast<int>(obs[t]) == j)
                    {
                        numer += gamma[t][i];
                    }
                    denom += gamma[t][i];
                }
                emit_p[i * M + j] = numer / denom;
            }
        }

        // Clean up alphas and betas from this iteration (they're reallocated each time by forward/backward)
        for (int t = 0; t <= T; t++)
        {
            delete[] alphas[t];
            delete[] betas[t];
        }
        delete[] alphas;
        delete[] betas;
    }

    // Clean up gamma and xi once after all iterations
    for (int t = 0; t < T; t++)
    {
        delete[] gamma[t];
        delete[] xi[t];
    }
    delete[] gamma;
    delete[] xi;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Baum-Welch algorithm execution time: " << format_duration(duration) << std::endl;
}

std::string IHMM::viterbi(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M)
{
    auto start = std::chrono::high_resolution_clock::now();

    // PARALLELIZATION STRATEGY FOR VITERBI ALGORITHM:
    // 1. SIMILAR TO FORWARD: Viterbi is like forward algorithm but uses max instead of sum
    // 2. THREAD MAPPING: N threads per time step, each computing one state
    // 3. PARALLEL MAX REDUCTION: Instead of parallel sum reduction, use parallel max reduction
    // 4. PATH TRACKING: Store backpointers in global memory, reconstruct path on CPU
    // 5. MEMORY OPTIMIZATION: Use shared memory for V matrix values within each time step

    float **V = new float *[N];
    int **path = new int *[N];

    for (int i = 0; i < N; i++)
    {
        V[i] = new float[T];
        path[i] = new int[T];
    }

    // Initialize first time step
    // CUDA: Parallel initialization with N threads
    for (int i = 0; i < N; i++)
    {
        V[i][0] = start_p[i] * emit_p[i * M + static_cast<int>(obs[0])];
        path[i][0] = -1;
    }

    // Recursive step
    // CUDA: Each time step requires synchronization, but states within time step are parallel
    for (int t = 1; t < T; t++)
    {
        // CUDA KERNEL: Launch N threads, each computing one current state
        for (int i = 0; i < N; i++)
        {
            float max_prob = 0;
            int max_state = 0;

            // CUDA: This loop can use parallel max reduction
            // Use shared memory and reduction to find maximum efficiently
            for (int j = 0; j < N; j++)
            {
                float prob = V[j][t - 1] * trans_p[j * N + i] * emit_p[i * M + static_cast<int>(obs[t])];
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_state = j;
                }
            }
            V[i][t] = max_prob;
            path[i][t] = max_state;
        }
        // CUDA: __syncthreads() needed before next time step
    }

    // Find most likely state at last time step
    // CUDA: Parallel max reduction across N final states
    float max_prob = 0;
    int max_state = 0;
    for (int i = 0; i < N; i++)
    {
        if (V[i][T - 1] > max_prob)
        {
            max_prob = V[i][T - 1];
            max_state = i;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // std::cout << "Viterbi algorithm execution time: " << format_duration(duration) << std::endl;

    int *optimal_path = new int[T];

    // Backtracking
    // CUDA: This is inherently sequential and should be done on CPU
    optimal_path[T - 1] = max_state;
    for (int t = T - 2; t >= 0; t--)
    {
        optimal_path[t] = path[optimal_path[t + 1]][t + 1];
    }

    std::ostringstream state_str;
    for (int i = 0; i < T; i++)
    {
        state_str << optimal_path[i];
        if (i < T - 1)
        {
            state_str << "";
        }
    }

    // std::cout << "Most likely path: " << state_str.str() << std::endl;

    // Clean up
    for (int i = 0; i < N; i++)
    {
        delete[] V[i];
        delete[] path[i];
    }
    delete[] V;
    delete[] path;

    return state_str.str();
}