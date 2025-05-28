#include "hmm.hpp"
#include <sstream>
#include <iostream>

void IHMM::forward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M) {
    // Empty
}

void IHMM::backward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M) {
    // Empty 
}

void IHMM::baum_welch(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M) {
    // Empty
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