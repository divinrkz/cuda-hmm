#ifndef HMM_GPU_CUH
#define HMM_GPU_CUH

#include <vector>
#include <string>

// A unified class to handle all HMM algorithms on the GPU.
// Manages device memory internally.
class HMM_GPU {
public:
    // Constructor initializes model parameters on the GPU.
    HMM_GPU(int num_states, int num_obs_symbols, int max_seq_len);

    // Destructor frees all GPU memory.
    ~HMM_GPU();

    /**
     * @brief Computes the log-likelihood of an observation sequence using the Forward algorithm.
     * @param h_obs Host pointer to the observation sequence.
     * @param h_A Host pointer to the transition matrix.
     * @param h_B Host pointer to the emission matrix.
     * @param h_pi Host pointer to the initial state probabilities.
     * @param T The actual length of the observation sequence.
     * @return The total log-likelihood P(O|lambda).
     */
    float forward(const int* h_obs, const float* h_A, const float* h_B, const float* h_pi, int T);

    /**
     * @brief Computes the most likely hidden state sequence using the Viterbi algorithm.
     * @param h_obs Host pointer to the observation sequence.
     * @param h_A Host pointer to the transition matrix.
     * @param h_B Host pointer to the emission matrix.
     * @param h_pi Host pointer to the initial state probabilities.
     * @param T The actual length of the observation sequence.
     * @return A string representing the optimal state sequence.
     */
    std::string viterbi(const int* h_obs, const float* h_A, const float* h_B, const float* h_pi, int T);

    /**
     * @brief Trains the HMM parameters using the Baum-Welch algorithm.
     * @param h_obs Host pointer to the observation sequence.
     * @param h_A Host pointer to the transition matrix (will be updated).
     * @param h_B Host pointer to the emission matrix (will be updated).
     * @param h_pi Host pointer to the initial state probabilities (will be updated).
     * @param T The actual length of the observation sequence.
     * @param max_iters Maximum number of training iterations.
     * @param tolerance Log-likelihood change for convergence.
     */
    void baum_welch(const int* h_obs, float* h_A, float* h_B, float* h_pi, int T, int max_iters, float tolerance);

private:
    // Internal helper functions to execute steps of the algorithms.
    void run_forward_pass_scan(int T);
    void run_backward_pass_scan(int T);
    void run_viterbi_pass_scan(int T);

    // Model and sequence parameters
    int N; // num_states
    int M; // num_obs_symbols
    int max_T; // max sequence length

    // --- Device Memory Pointers ---
    float* d_A;      // Transition matrix
    float* d_B;      // Emission matrix
    float* d_pi;     // Initial state probabilities
    int*   d_obs;    // Observation sequence

    float* d_alpha;  // Forward probabilities
    float* d_beta;   // Backward probabilities
    float* d_gamma;  // State occupancy probabilities
    float* d_xi;     // State transition probabilities
    float* d_log_likelihood;

    // Viterbi-specific memory
    float* d_viterbi_probs; // v_t(j)
    int*   d_viterbi_paths; // backpointers

    // Memory for parallel scans
    float* d_scan_M;     // Input matrices for scan (T x N x N)
    float* d_scan_P;     // Output prefix-sum matrices (T x N x N)
    float* d_scan_temp;  // Temporary buffer for scan
};

#endif // HMM_GPU_CUH