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
    
    void initializeDeviceMemory(int max_T, int num_states, int num_observations);
    void freeDeviceMemory();
    void convertToLogSpace(float* probs, float* log_probs, int size);
    void normalizeArray(float* array, int size);

public:
    HMMCuda(int num_states, int num_observations);
    ~HMMCuda();
    
    float **forward(float *obs, int *states, float *start_p, float *trans_p, 
                   float *emit_p, int T, int N, int M) override;
    float **backward(float *obs, int *states, float *start_p, float *trans_p,
                    float *emit_p, int T, int N, int M) override;
    std::string viterbi(float *obs, int *states, float *start_p, float *trans_p,
                       float *emit_p, int T, int N, int M) override;
    void baum_welch(float *obs, int *states, float *start_p, float *trans_p,
                    float *emit_p, int T, int N, int M, int N_iters) override;
    void forward_backward(float *obs, int *states, float *start_p, float *trans_p,
                         float *emit_p, int T, int N, int M) override;
};