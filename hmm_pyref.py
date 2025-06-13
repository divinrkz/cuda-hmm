"""
HMM Reference Implementation
Based on CS155 Set 6 solution, adapted for configuration file testing
"""

import numpy as np
import random
import argparse
import sys
import os

# Set random seed for reproducibility
np.random.seed(seed=123)


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.
        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
        Parameters:
            L:          Number of states.
            D:          Number of observations.
            A:          The transition matrix.
            O:          The observation matrix.
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            seqs[1][i] = str(i)

        for d in range(2, M + 1):
            for curr_state in range(self.L):
                max_prob = float("-inf")
                best_seq = ""

                for prev_state in range(self.L):
                    prob = probs[d-1][prev_state] * self.A[prev_state][curr_state] * self.O[curr_state][x[d-1]]
                    if prob >= max_prob:
                        max_prob = prob
                        best_seq = seqs[d-1][prev_state] + str(curr_state)

                probs[d][curr_state] = max_prob
                seqs[d][curr_state] = best_seq

        max_seq = seqs[M][np.argmax(probs[M])]
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.
                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]

        for d in range(2, M + 1):
            for curr_state in range(self.L):
                prob = 0
                for prev_state in range(self.L):
                    prob += (self.O[curr_state][x[d-1]] * (alphas[d-1][prev_state] * self.A[prev_state][curr_state]))

                alphas[d][curr_state] = prob

            if normalize:
                denom = np.sum(alphas[d])
                alphas[d] = [alpha/denom for alpha in alphas[d]]

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            betas[M][i] = 1

        for d in range(M - 1, -1, -1):
            for curr_state in range(self.L):
                prob = 0
                for next_state in range(self.L):
                    if d == 0:
                        prob += (betas[d+1][next_state] * self.A_start[next_state] * self.O[next_state][x[d]])
                    else:
                        prob += (betas[d+1][next_state] * self.A[curr_state][next_state] * self.O[next_state][x[d]])

                betas[d][curr_state] = prob

            if normalize:
                denom = np.sum(betas[d])
                betas[d] = [beta/denom for beta in betas[d]]

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        for prev_state in range(self.L):
            for curr_state in range(self.L):
                numer_array = []
                denom_array = []

                for x in range(len(X)):
                    for y in range(1, len(Y[x])):
                        if Y[x][y] == curr_state and Y[x][y-1] == prev_state:
                            numer_array.append(1)
                        else:
                            numer_array.append(0)

                        if Y[x][y-1] == prev_state:
                            denom_array.append(1)
                        else:
                            denom_array.append(0)

                numer = np.sum(numer_array)
                denom = np.sum(denom_array)
                self.A[prev_state][curr_state] = numer/denom

        # Calculate each element of O using the M-step formulas.
        for curr_state in range(self.L):
            for curr_obs in range(self.D):
                numer_array = []
                denom_array = []

                for x in range(len(X)):
                    for y in range(len(Y[x])):
                        if X[x][y] == curr_obs and Y[x][y] == curr_state:
                            numer_array.append(1)
                        else:
                            numer_array.append(0)

                        if Y[x][y] == curr_state:
                            denom_array.append(1)
                        else:
                            denom_array.append(0)

                numer = np.sum(numer_array)
                denom = np.sum(denom_array)
                self.O[curr_state][curr_obs] = numer/denom

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of variable-length lists, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            N_iters:    The number of iterations to train on.
        '''

        for i in range(N_iters):
            A_numer = np.zeros((self.L, self.L))
            A_denom = np.zeros((self.L, self.L))
            O_numer = np.zeros((self.L, self.D))
            O_denom = np.zeros((self.L, self.D))

            for x in X:
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)
                M = len(x)

                for d in range(1, M + 1):
                    prob_OAd = np.array([alphas[d][curr_state] * betas[d][curr_state] for curr_state in range(self.L)])
                    prob_OAd /= np.sum(prob_OAd)

                    for curr_state in range(self.L):
                        O_numer[curr_state][x[d-1]] += prob_OAd[curr_state]
                        O_denom[curr_state] += prob_OAd[curr_state]
                        if d != M:
                            A_denom[curr_state] += prob_OAd[curr_state]

                for d in range(1, M):
                    prob_An = np.array([[alphas[d][curr_state] \
                                        * self.O[next_state][x[d]] \
                                        * self.A[curr_state][next_state] \
                                        * betas[d+1][next_state] \
                                        for next_state in range(self.L)] \
                                        for curr_state in range(self.L)])
                    prob_An /= np.sum(prob_An)

                    for curr_state in range(self.L):
                        for next_state in range(self.L):
                            A_numer[curr_state][next_state] += prob_An[curr_state][next_state]

            self.A = A_numer / A_denom
            self.O = O_numer / O_denom

    def generate_emission(self, M, seed=None):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        # (Re-)Initialize random number generator
        rng = np.random.default_rng(seed=seed)

        emission = []
        states = []

        # Initialize Random Start State
        state = np.random.randint(0, self.L)

        for d in range(M):
            emission.append(np.random.choice(list(range(self.D)), p = self.O[state]))
            states.append(state)
            state = np.random.choice(list(range(self.L)), p = self.A[state])

        return emission, states

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def parse_config_file(config_file):
    """
    Parse HMM configuration file in the format specified by the chmm project.
    Returns: (N, M, start_p, trans_p, emit_p, sequences)
    """
    with open(config_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
    
    line_idx = 0
    
    # Number of states
    N = int(lines[line_idx])
    line_idx += 1
    
    # Number of outputs
    M = int(lines[line_idx])
    line_idx += 1
    
    # Initial state probabilities
    start_p = [float(x) for x in lines[line_idx].split()]
    line_idx += 1
    
    # State transition probabilities (N x N matrix)
    trans_p = []
    for i in range(N):
        row = [float(x) for x in lines[line_idx].split()]
        trans_p.append(row)
        line_idx += 1
    
    # State output probabilities (N x M matrix)
    emit_p = []
    for i in range(N):
        row = [float(x) for x in lines[line_idx].split()]
        emit_p.append(row)
        line_idx += 1
    
    # Data size and length
    data_info = [int(x) for x in lines[line_idx].split()]
    num_sequences = data_info[0]
    sequence_length = data_info[1]
    line_idx += 1
    
    # Parse sequences
    sequences = []
    remaining_data = ' '.join(lines[line_idx:])
    obs_data = [int(x) for x in remaining_data.split()]
    
    for i in range(num_sequences):
        start_idx = i * sequence_length
        end_idx = start_idx + sequence_length
        sequences.append(obs_data[start_idx:end_idx])
    
    return N, M, start_p, trans_p, emit_p, sequences


def run_forward_algorithm(config_file):
    """Run forward algorithm (problem 1) and return probabilities"""
    N, M, start_p, trans_p, emit_p, sequences = parse_config_file(config_file)
    
    # Create HMM
    hmm = HiddenMarkovModel(trans_p, emit_p)
    hmm.A_start = start_p
    
    results = []
    for seq in sequences:
        prob = hmm.probability_alphas(seq)
        results.append(prob)
    
    return results


def run_viterbi_algorithm(config_file):
    """Run Viterbi algorithm (problem 2) and return most probable sequences"""
    N, M, start_p, trans_p, emit_p, sequences = parse_config_file(config_file)
    
    # Create HMM
    hmm = HiddenMarkovModel(trans_p, emit_p)
    hmm.A_start = start_p
    
    results = []
    for seq in sequences:
        path = hmm.viterbi(seq)
        results.append(path)
    
    return results


def run_baum_welch_algorithm(config_file, n_iters=100):
    """Run Baum-Welch algorithm (problem 3) for training"""
    N, M, start_p, trans_p, emit_p, sequences = parse_config_file(config_file)
    
    # Create HMM with initial parameters
    hmm = HiddenMarkovModel(trans_p, emit_p)
    hmm.A_start = start_p
    
    # Train using Baum-Welch
    hmm.unsupervised_learning(sequences, n_iters)
    
    return hmm.A_start, hmm.A, hmm.O


def main():
    parser = argparse.ArgumentParser(description='HMM Reference Implementation')
    parser.add_argument('config_file', help='Path to HMM configuration file')
    parser.add_argument('--problem', '-p', choices=['1', '2', '3'], required=True,
                      help='Problem to solve: 1=forward, 2=viterbi, 3=baum-welch')
    parser.add_argument('--iterations', '-n', type=int, default=100,
                      help='Number of iterations for Baum-Welch (problem 3)')
    parser.add_argument('--output', '-o', help='Output file for results')
    
    args = parser.parse_args()
    
    if args.problem == '1':
        # Forward algorithm - compute probabilities
        results = run_forward_algorithm(args.config_file)
        print("Forward Algorithm Results:")
        for i, prob in enumerate(results):
            print(f"Sequence {i+1}: {prob:.6e}")
            
    elif args.problem == '2':
        # Viterbi algorithm - most probable sequences
        results = run_viterbi_algorithm(args.config_file)
        print("Viterbi Algorithm Results:")
        for i, path in enumerate(results):
            print(f"Sequence {i+1}: {path}")
            
    elif args.problem == '3':
        # Baum-Welch algorithm - parameter training
        start_p, trans_p, emit_p = run_baum_welch_algorithm(args.config_file, args.iterations)
        print("Baum-Welch Algorithm Results:")
        print("Initial probabilities:")
        print(' '.join(f"{p:.6f}" for p in start_p))
        print("Transition matrix:")
        for row in trans_p:
            print(' '.join(f"{p:.6f}" for p in row))
        print("Emission matrix:")
        for row in emit_p:
            print(' '.join(f"{p:.6f}" for p in row))
    
    # Save results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            if args.problem == '1':
                for prob in results:
                    f.write(f"{prob:.6e}\n")
            elif args.problem == '2':
                for path in results:
                    f.write(f"{path}\n")
            elif args.problem == '3':
                f.write("# Initial probabilities\n")
                f.write(' '.join(f"{p:.6f}" for p in start_p) + "\n")
                f.write("# Transition matrix\n")
                for row in trans_p:
                    f.write(' '.join(f"{p:.6f}" for p in row) + "\n")
                f.write("# Emission matrix\n")
                for row in emit_p:
                    f.write(' '.join(f"{p:.6f}" for p in row) + "\n")


if __name__ == "__main__":
    main() 