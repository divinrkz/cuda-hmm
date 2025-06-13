#pragma once
#include "hmm.hpp"
#include <string>

class HMMCuda : public IHMM {
private:
    // Device memory pointers
    float *d_alpha, *d_beta, *d_gamma, *d_xi;
    float *d_V_curr, *d_V_prev;
    float *d_log_trans_p, *d_log_emit_p, *d_log_start_p;
    int *d_path, *d_obs;
    
    // Host memory for results
    float **h_alpha, **h_beta;
    int max_T_allocated;
    int current_T_allocated;  // Track current allocation size
    
    void initializeFixedMemory(int num_states, int num_observations);
    void ensureSequenceMemory(int T, int N, int M);
    void ensureBaumWelchMemory(int T, int N, int M);
    void freeSequenceMemory();
    void freeFixedMemory();
    
    void convertToLogSpace(float* probs, float* log_probs, int size);
    void normalizeArray(float* array, int size);
    void copyObservationsToDevice(float* obs, int T);

public:
    HMMCuda(int num_states, int num_observations);
    ~HMMCuda();
    
    float **forward(float *obs, int *states, float *start_p, float *trans_p, 
                   float *emit_p, int T, int N, int M);
    float **backward(float *obs, int *states, float *start_p, float *trans_p,
                    float *emit_p, int T, int N, int M);
    std::string viterbi(float *obs, int *states, float *start_p, float *trans_p,
                       float *emit_p, int T, int N, int M);
    void baum_welch(float *obs, int *states, float *start_p, float *trans_p,
                    float *emit_p, int T, int N, int M, int N_iters);
    void forward_backward(float *obs, int *states, float *start_p, float *trans_p,
                         float *emit_p, int T, int N, int M);
};