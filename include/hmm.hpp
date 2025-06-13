#pragma once
#include <string>

class IHMM
{
private:
    int N;          // number of states
    int M;          // number of observations
    int T;          // number of time steps
    int *states;    // array of states
    float *start_p; // initial state probabilities
    float *trans_p; // transition probabilities
    float *emit_p;  // emission probabilities

public:
    IHMM(int num_states, int num_observations) : N(num_states),
                                                 M(num_observations),
                                                 states(new int[N]),
                                                 start_p(new float[N]),
                                                 trans_p(new float[N * N]),
                                                 emit_p(new float[N * M]) {}

    virtual ~IHMM()
    {
        delete[] states;
        delete[] start_p;
        delete[] trans_p;
        delete[] emit_p;
    }

    virtual float **forward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M);
    virtual float **backward(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M);
    virtual std::string viterbi(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M);
    virtual void baum_welch(float *obs, int *states, float *start_p, float *trans_p, float *emit_p, int T, int N, int M, int N_iters);
};