// File: hmm_gpu.cuh
#ifndef HMM_GPU_CUH
#define HMM_GPU_CUH

#include <vector>
#include <string>

// A class implementing the state-wise parallel HMM algorithms.
class HMM_GPU
{
public:
    HMM_GPU(int num_states, int num_obs_symbols, int max_seq_len);
    ~HMM_GPU();

    float forward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T);
    float backward(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T);
    std::string viterbi(const int *h_obs, const float *h_A, const float *h_B, const float *h_pi, int T);
    void baum_welch(const int *h_obs, float *h_A, float *h_B, float *h_pi, int T, int max_iters, float tolerance = 1e-5f);

private:
    void _forward_internal(const int *h_obs, int T);
    void _backward_internal(const int *h_obs, int T);

    int N, M, max_T;

    // Device Memory Pointers
    float *d_A;
    float *d_B;
    float *d_pi;
    int *d_obs;
    float *d_alpha;
    float *d_beta;
    float *d_gamma;
    float *d_xi;
    float *d_viterbi_probs;
    int *d_viterbi_paths;
};

#endif // HMM_GPU_CUH