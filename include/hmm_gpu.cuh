#ifndef HMM_GPU_CUH
#define HMM_GPU_CUH

#include <vector>
#include <string>

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
    void run_forward_pass_internal(int T, const int* h_obs_for_indexing);
    void run_backward_pass_internal(int T, const int* h_obs_for_indexing);
    void run_forward_pass_scaled(int T, const int* h_obs_for_indexing);
    void run_backward_pass_scaled(int T, const int* h_obs_for_indexing);

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
    float *d_scaling_factors;
};

#endif 