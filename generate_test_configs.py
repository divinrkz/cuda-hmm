"""
Configuration file generator for HMM testing
Generates test configuration files in the chmm format
"""

import numpy as np
import argparse
import os
import random

def generate_random_probabilities(size, seed=None):
    """Generate random probabilities that sum to 1.0"""
    if seed is not None:
        np.random.seed(seed)
    
    probs = np.random.random(size)
    return probs / np.sum(probs)

def generate_random_matrix(rows, cols, seed=None):
    """Generate random probability matrix where each row sums to 1.0"""
    if seed is not None:
        np.random.seed(seed)
    
    matrix = []
    for i in range(rows):
        row = generate_random_probabilities(cols, seed=(seed+i if seed else None))
        matrix.append(row)
    return matrix

def generate_simple_hmm_config(n_states, n_obs, seq_length, n_sequences, filename, seed=42):
    """
    Generate a simple HMM configuration file for testing
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate initial state probabilities (uniform for simplicity)
    start_p = [1.0 / n_states] * n_states
    
    # Generate transition matrix
    trans_p = generate_random_matrix(n_states, n_states, seed)
    
    # Generate emission matrix  
    emit_p = generate_random_matrix(n_states, n_obs, seed)
    
    # Generate observation sequences
    sequences = []
    for i in range(n_sequences):
        seq = [random.randint(0, n_obs-1) for _ in range(seq_length)]
        sequences.append(seq)
    
    # Write configuration file
    with open(filename, 'w') as f:
        f.write("# Simple HMM test configuration\n")
        f.write(f"# {n_states} states, {n_obs} observations\n")
        f.write(f"# {n_sequences} sequences of length {seq_length}\n")
        f.write("\n")
        
        # Number of states
        f.write(f"{n_states}\n")
        
        # Number of observation symbols
        f.write(f"{n_obs}\n")
        
        # Initial state probabilities
        f.write(" ".join(f"{p:.6f}" for p in start_p) + "\n")
        
        # Transition matrix
        for row in trans_p:
            f.write(" ".join(f"{p:.6f}" for p in row) + "\n")
        
        # Emission matrix
        for row in emit_p:
            f.write(" ".join(f"{p:.6f}" for p in row) + "\n")
        
        # Data size and length
        f.write(f"{n_sequences} {seq_length}\n")
        
        # Observation sequences (each sequence on its own line)
        for seq in sequences:
            f.write(" ".join(str(obs) for obs in seq) + "\n")

def generate_coin_flip_hmm(filename):
    """
    Generate a classic coin flip HMM example
    2 states: Fair coin (F) and Biased coin (B)
    2 observations: Heads (0) and Tails (1)
    """
    
    with open(filename, 'w') as f:
        f.write("# Classic coin flip HMM\n")
        f.write("# 2 states: Fair coin (0) and Biased coin (1)\n")
        f.write("# 2 observations: Heads (0) and Tails (1)\n")
        f.write("\n")
        
        # 2 states
        f.write("2\n")
        
        # 2 observation symbols
        f.write("2\n")
        
        # Initial state probabilities (equal chance of starting with either coin)
        f.write("0.5 0.5\n")
        
        # Transition matrix
        # Fair coin tends to stay fair, biased coin tends to stay biased
        f.write("0.7 0.3\n")  # From fair coin
        f.write("0.4 0.6\n")  # From biased coin
        
        # Emission matrix
        # Fair coin: 50/50 chance of heads/tails
        # Biased coin: 80% chance of heads, 20% chance of tails
        f.write("0.5 0.5\n")  # Fair coin emissions
        f.write("0.8 0.2\n")  # Biased coin emissions
        
        # Data: 5 sequences of length 10
        f.write("5 10\n")
        
        # Sample observation sequences (manually created for predictable testing)
        sequences = [
            [0, 0, 1, 0, 1, 0, 0, 1, 1, 0],  # Mixed
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # Mostly heads (likely biased)
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],  # Alternating
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Very head-heavy (likely biased)
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 1]   # Mostly tails
        ]
        
        # Observation sequences (each sequence on its own line)
        for seq in sequences:
            f.write(" ".join(str(obs) for obs in seq) + "\n")

def generate_weather_hmm(filename):
    """
    Generate a weather prediction HMM
    3 states: Sunny (0), Cloudy (1), Rainy (2)
    2 observations: Dry (0), Wet (1)
    """
    
    with open(filename, 'w') as f:
        f.write("# Weather prediction HMM\n")
        f.write("# 3 states: Sunny (0), Cloudy (1), Rainy (2)\n")
        f.write("# 2 observations: Dry (0), Wet (1)\n")
        f.write("\n")
        
        # 3 states
        f.write("3\n")
        
        # 2 observation symbols  
        f.write("2\n")
        
        # Initial state probabilities
        f.write("0.6 0.3 0.1\n")  # More likely to start sunny
        
        # Transition matrix
        f.write("0.7 0.2 0.1\n")  # From sunny
        f.write("0.3 0.4 0.3\n")  # From cloudy
        f.write("0.2 0.3 0.5\n")  # From rainy
        
        # Emission matrix
        f.write("0.9 0.1\n")  # Sunny: mostly dry
        f.write("0.6 0.4\n")  # Cloudy: somewhat dry
        f.write("0.1 0.9\n")  # Rainy: mostly wet
        
        # Data: 3 sequences of length 15
        f.write("3 15\n")
        
        # Sample observation sequences
        sequences = [
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        ]
        
        # Observation sequences (each sequence on its own line)
        for seq in sequences:
            f.write(" ".join(str(obs) for obs in seq) + "\n")

def create_test_suite():
    """Create a comprehensive test suite with various HMM configurations"""
    
    # Create test directory
    test_dir = "test_configs"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Creating test configurations in {test_dir}/")
    
    # Simple test cases
    generate_simple_hmm_config(2, 2, 5, 3, f"{test_dir}/simple_2x2.cfg", seed=42)
    generate_simple_hmm_config(3, 3, 8, 4, f"{test_dir}/simple_3x3.cfg", seed=123)
    generate_simple_hmm_config(4, 2, 10, 5, f"{test_dir}/simple_4x2.cfg", seed=456)
    
    # Classic examples
    generate_coin_flip_hmm(f"{test_dir}/coin_flip.cfg")
    generate_weather_hmm(f"{test_dir}/weather.cfg")
    
    # Copy the example from the chmm repository
    create_chmm_example(f"{test_dir}/chmm_example.cfg")
    
    print("Test configurations created:")
    for filename in os.listdir(test_dir):
        if filename.endswith('.cfg'):
            print(f"  - {filename}")

def create_chmm_example(filename):
    """Create the example configuration from the chmm repository README"""
    
    with open(filename, 'w') as f:
        f.write("# HMM model configuration from chmm repository example\n")
        f.write("# 16 states with 2 observation symbols and 32 input sequences\n")
        f.write("\n")
        
        # 16 states
        f.write("16\n")
        
        # 2 observation symbols
        f.write("2\n")
        
        # Initial state probabilities
        start_probs = [0.04, 0.02, 0.06, 0.04, 0.11, 0.11, 0.01, 0.09, 
                      0.03, 0.05, 0.06, 0.11, 0.05, 0.11, 0.03, 0.08]
        f.write(" ".join(f"{p:.2f}" for p in start_probs) + "\n")
        
        # Transition matrix (16x16) - simplified version
        np.random.seed(42)
        for i in range(16):
            row = generate_random_probabilities(16)
            f.write(" ".join(f"{p:.2f}" for p in row) + "\n")
        
        # Emission matrix (16x2)
        np.random.seed(123)
        for i in range(16):
            row = generate_random_probabilities(2)
            f.write(" ".join(f"{p:.2f}" for p in row) + "\n")
        
        # Data: 5 sequences of length 10 (simplified from original 32x10)
        f.write("5 10\n")
        
        # Sample sequences
        np.random.seed(789)
        for i in range(5):
            seq = [np.random.randint(0, 2) for _ in range(10)]
            f.write(" ".join(str(obs) for obs in seq) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Generate HMM test configuration files')
    parser.add_argument('--type', choices=['simple', 'coin', 'weather', 'chmm', 'all'], 
                       default='all', help='Type of configuration to generate')
    parser.add_argument('--output', '-o', help='Output filename')
    parser.add_argument('--states', '-s', type=int, default=3, help='Number of states (for simple type)')
    parser.add_argument('--obs', '-m', type=int, default=2, help='Number of observation symbols')
    parser.add_argument('--length', '-l', type=int, default=10, help='Sequence length')
    parser.add_argument('--sequences', '-n', type=int, default=5, help='Number of sequences')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.type == 'all':
        create_test_suite()
    elif args.type == 'simple':
        filename = args.output or f"simple_{args.states}x{args.obs}.cfg"
        generate_simple_hmm_config(args.states, args.obs, args.length, 
                                  args.sequences, filename, args.seed)
        print(f"Generated simple HMM configuration: {filename}")
    elif args.type == 'coin':
        filename = args.output or "coin_flip.cfg"
        generate_coin_flip_hmm(filename)
        print(f"Generated coin flip HMM configuration: {filename}")
    elif args.type == 'weather':
        filename = args.output or "weather.cfg"
        generate_weather_hmm(filename)
        print(f"Generated weather HMM configuration: {filename}")
    elif args.type == 'chmm':
        filename = args.output or "chmm_example.cfg"
        create_chmm_example(filename)
        print(f"Generated chmm example configuration: {filename}")

if __name__ == "__main__":
    main() 