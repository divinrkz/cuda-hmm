#include "hmm.hpp"
#include <sstream>
#include <iostream>

float** IHMM::forward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M) {
    // Create and initialize alphas (T+1 x N matrix)
    float** alphas = new float*[T + 1];
    for (int t = 0; t <= T; t++) {
        alphas[t] = new float[N](); // initialize with zeros
    }

    // Initialize first time step
    for (int i = 0; i < N; i++) {
        alphas[1][i] = start_p[i] * emit_p[i * M + static_cast<int>(obs[0])];
    }

    // Recursive step
    for (int t = 2; t <= T; t++) {
        for (int curr_state = 0; curr_state < N; curr_state++) {
            float prob = 0.0f;
            for (int prev_state = 0; prev_state < N; prev_state++) {
                prob += emit_p[curr_state * M + static_cast<int>(obs[t-1])] * 
                        (alphas[t-1][prev_state] * trans_p[prev_state * N + curr_state]);
            }
            alphas[t][curr_state] = prob;
        }
    }
    return alphas;
}

float** IHMM::backward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M) {
    // Create and initialize betas (T+1 x N matrix)
    float** betas = new float*[T + 1];
    for (int t = 0; t <= T; t++) {
        betas[t] = new float[N](); // initialize with zeros
    }

    // Initialize last time step
    for (int i = 0; i < N; i++) {
        betas[T][i] = 1.0f;
    }

    // Recursive step
    for (int t = T - 1; t >= 0; t--) {
        for (int curr_state = 0; curr_state < N; curr_state++) {
            float prob = 0.0f;
            for (int next_state = 0; next_state < N; next_state++) {
                prob += emit_p[next_state * M + static_cast<int>(obs[t+1])] * 
                        (betas[t+1][next_state] * trans_p[curr_state * N + next_state]);
            }
            betas[t][curr_state] = prob;
        }
    }

    return betas;

}

void IHMM::baum_welch(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M, int N_iters) {
    // assumes start_p, trans_p, emit_p are randomly initialized
    for (int iter = 0; iter < N_iters; iter++) {
        // ===== Expectation Step =====
        // Calculate forward and backward probabilities
        float** alphas = forward(obs, states, start_p, trans_p, emit_p, T, N, M);
        float** betas = backward(obs, states, start_p, trans_p, emit_p, T, N, M);

        // initialize gamma
        float** gamma = new float*[T];
        for (int t = 0; t < T; t++) {
            gamma[t] = new float[N]();  
        }

        // Compute gamma (probability of being in state i at time t)
        for (int t = 0; t < T; t++) {
            float denom = 0.0f;
            for (int i = 0; i < N; i++) {
                denom += alphas[t][i] * betas[t][i];
            }
            for (int i = 0; i < N; i++) {
                gamma[t][i] = alphas[t][i] * betas[t][i] / denom;
            }
        }

        // initialize xi
        float** xi = new float*[T];
        for (int t = 0; t < T; t++) {
            xi[t] = new float[N * N]();
        }

        // Compute xi (probability of being in state i at t and state j at t+1)
        for (int t = 0; t < T-1; t++) {
            float denom = 0.0f;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    xi[t][i * N + j] = alphas[t][i] * trans_p[i * N + j] * emit_p[j * M + static_cast<int>(obs[t+1])] * betas[t+1][j];
                    denom += xi[t][i * N + j];
                }
            }
            // Normalize xi
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    xi[t][i * N + j] /= denom;
                }
            }
        }

        // ===== Maximization Step =====
        // Update model parameters using computed expectations

        // Update initial state probabilities
        for (int i = 0; i < N; i++) {
            start_p[i] = gamma[0][i];
        }

        // Update transition probabilities
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float numer = 0.0f;
                float denom = 0.0f;
                for (int t = 0; t < T-1; t++) {
                    numer += xi[t][i * N + j];
                    denom += gamma[t][i];
                }
                trans_p[i * N + j] = numer / denom;
            }
        }

        // Update emission probabilities
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                float numer = 0.0f;
                float denom = 0.0f;
                for (int t = 0; t < T; t++) {
                    if (static_cast<int>(obs[t]) == j) {
                        numer += gamma[t][i];
                    }
                    denom += gamma[t][i];
                }
                emit_p[i * M + j] = numer / denom;
            }
        }
    }

    // Clean up
    for (int t = 0; t < T; t++) {
        delete[] alphas[t];
        delete[] betas[t];
        delete[] gamma[t];
        delete[] xi[t];
    }
    delete[] alphas;
    delete[] betas;
    delete[] gamma;
    delete[] xi;
}

void IHMM::forward_backward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M) {
    // Empty
}

std::string IHMM::viterbi(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M) {
    float** V = new float*[N];
    int **path = new int*[N];

    for(int i = 0; i < N; i++) {
        V[i] = new float[T];
        path[i] = new int[T];
    }

    for (int i = 0; i < N; i++) {
        V[i][0] = start_p[i] * emit_p[i * M + static_cast<int>(obs[0])];
        path[i][0] = -1;
    }

    
    for (int t = 1; t < T; t++) {
        for (int i = 0; i < N; i++) {
            float max_prob = 0;
            int max_state = 0;

            for (int j = 0; j < N; j++) {
                float prob = V[j][t - 1] * trans_p[j * N + i] * emit_p[i * M + static_cast<int>(obs[t])];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_state = j;
                }
            }
            V[i][t] = max_prob;
            path[i][t] = max_state;
        }
    }


    float max_prob = 0;
    int max_state = 0;
    for (int i = 0; i < N; i++) {
        if (V[i][T - 1] > max_prob) {
            max_prob = V[i][T - 1];
            max_state = i;
        }
    }


    int* optimal_path = new int[T];
    
    optimal_path[T - 1] = max_state;
    for (int t = T - 2; t >= 0; t--) {
        optimal_path[t] = path[optimal_path[t + 1]][t + 1];
    }

    std::ostringstream state_str;
    for (int i = 0; i < T; i++) {
        state_str << optimal_path[i];
        if (i < T - 1) {
            state_str << " -> ";
        }
    }

    std::cout << "Most likely path: " << state_str.str() << std::endl;

    // Clean up
    for(int i = 0; i < N; i++) {
        delete[] V[i];
        delete[] path[i];
    }
    delete[] V;
    delete[] path;

    return state_str.str();
} 