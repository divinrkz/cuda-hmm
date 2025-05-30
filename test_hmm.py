"""
HMM Test Runner
Compares output between Python reference implementation and C++ implementation
"""

import os
import sys
import subprocess
import argparse
import time
import numpy as np
from hmm_pyref import run_forward_algorithm, run_viterbi_algorithm, run_baum_welch_algorithm, HiddenMarkovModel, parse_config_file

def run_cpp_hmm(config_file, problem, iterations=100, hmm_executable="./hmm"):
    """
    Run the C++ HMM implementation
    """
    if not os.path.exists(hmm_executable):
        raise FileNotFoundError(f"C++ HMM executable not found: {hmm_executable}")
    
    cmd = [hmm_executable, "-c", config_file, f"-p{problem}"]
    if problem == "3":
        cmd.extend(["-n", str(iterations)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"C++ HMM failed: {result.stderr}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("C++ HMM execution timed out")

def parse_cpp_forward_output(output):
    """Parse C++ forward algorithm output to extract probabilities"""
    lines = output.split('\n')
    probabilities = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            try:
                prob = float(line)
                probabilities.append(prob)
            except ValueError:
                continue
    
    return probabilities

def parse_cpp_viterbi_output(output):
    """Parse C++ Viterbi output to extract state sequences"""
    lines = output.split('\n')
    sequences = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Check if line contains only digits (state sequence)
            if line.replace(' ', '').isdigit():
                sequences.append(line.replace(' ', ''))
    
    return sequences

def parse_cpp_baum_welch_output(output):
    """Parse C++ Baum-Welch output to extract learned parameters"""
    lines = output.split('\n')
    start_p = None
    trans_p = []
    emit_p = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
            
        # First non-comment line should be initial probabilities
        if start_p is None:
            start_p = [float(x) for x in line.split()]
            i += 1
            continue
        
        # Try to parse as matrix rows
        try:
            row = [float(x) for x in line.split()]
            if trans_p == [] or len(trans_p) < len(start_p):
                trans_p.append(row)
            else:
                emit_p.append(row)
        except ValueError:
            pass
        
        i += 1
    
    return start_p, trans_p, emit_p

def compare_probabilities(python_probs, cpp_probs, tolerance=1e-6):
    """Compare probability outputs with given tolerance"""
    if len(python_probs) != len(cpp_probs):
        return False, f"Length mismatch: Python {len(python_probs)} vs C++ {len(cpp_probs)}"
    
    for i, (p_prob, c_prob) in enumerate(zip(python_probs, cpp_probs)):
        if abs(p_prob - c_prob) > tolerance:
            return False, f"Probability mismatch at index {i}: Python {p_prob:.6e} vs C++ {c_prob:.6e}"
    
    return True, "Probabilities match within tolerance"

def compare_sequences(python_seqs, cpp_seqs):
    """Compare Viterbi sequence outputs"""
    if len(python_seqs) != len(cpp_seqs):
        return False, f"Length mismatch: Python {len(python_seqs)} vs C++ {len(cpp_seqs)}"
    
    for i, (p_seq, c_seq) in enumerate(zip(python_seqs, cpp_seqs)):
        if p_seq != c_seq:
            return False, f"Sequence mismatch at index {i}: Python '{p_seq}' vs C++ '{c_seq}'"
    
    return True, "Sequences match exactly"

def compare_matrices(python_matrix, cpp_matrix, tolerance=1e-6, name="Matrix"):
    """Compare matrix outputs"""
    if python_matrix is None or cpp_matrix is None:
        return False, f"{name} is None"
    
    if len(python_matrix) != len(cpp_matrix):
        return False, f"{name} row count mismatch: Python {len(python_matrix)} vs C++ {len(cpp_matrix)}"
    
    for i, (p_row, c_row) in enumerate(zip(python_matrix, cpp_matrix)):
        if len(p_row) != len(c_row):
            return False, f"{name} column count mismatch at row {i}: Python {len(p_row)} vs C++ {len(c_row)}"
        
        for j, (p_val, c_val) in enumerate(zip(p_row, c_row)):
            if abs(p_val - c_val) > tolerance:
                return False, f"{name} mismatch at [{i}][{j}]: Python {p_val:.6f} vs C++ {c_val:.6f}"
    
    return True, f"{name} matches within tolerance"

def test_forward_algorithm(config_file, hmm_executable="./hmm"):
    """Test forward algorithm (problem 1)"""
    print(f"Testing Forward Algorithm with {config_file}")
    
    # Run Python implementation
    python_probs = run_forward_algorithm(config_file)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "1", hmm_executable=hmm_executable)
    cpp_probs = parse_cpp_forward_output(cpp_output)
    
    # Compare results
    match, message = compare_probabilities(python_probs, cpp_probs)
    
    print(f"  Python results: {[f'{p:.6e}' for p in python_probs]}")
    print(f"  C++ results:    {[f'{p:.6e}' for p in cpp_probs]}")
    print(f"  Match: {'✓' if match else '✗'} - {message}")
    
    return match

def test_viterbi_algorithm(config_file, hmm_executable="./hmm"):
    """Test Viterbi algorithm (problem 2)"""
    print(f"Testing Viterbi Algorithm with {config_file}")
    
    # Run Python implementation
    python_seqs = run_viterbi_algorithm(config_file)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "2", hmm_executable=hmm_executable)
    cpp_seqs = parse_cpp_viterbi_output(cpp_output)
    
    # Compare results
    match, message = compare_sequences(python_seqs, cpp_seqs)
    
    print(f"  Python results: {python_seqs}")
    print(f"  C++ results:    {cpp_seqs}")
    print(f"  Match: {'✓' if match else '✗'} - {message}")
    
    return match

def test_baum_welch_algorithm(config_file, iterations=10, hmm_executable="./hmm"):
    """Test Baum-Welch algorithm (problem 3)"""
    print(f"Testing Baum-Welch Algorithm with {config_file} ({iterations} iterations)")
    
    # Run Python implementation
    python_start, python_trans, python_emit = run_baum_welch_algorithm(config_file, iterations)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "3", iterations, hmm_executable=hmm_executable)
    cpp_start, cpp_trans, cpp_emit = parse_cpp_baum_welch_output(cpp_output)
    
    # Compare results
    start_match, start_msg = compare_matrices([python_start], [cpp_start], name="Initial probabilities")
    trans_match, trans_msg = compare_matrices(python_trans, cpp_trans, name="Transition matrix")
    emit_match, emit_msg = compare_matrices(python_emit, cpp_emit, name="Emission matrix")
    
    overall_match = start_match and trans_match and emit_match
    
    print(f"  Initial probs: {'✓' if start_match else '✗'} - {start_msg}")
    print(f"  Transition:    {'✓' if trans_match else '✗'} - {trans_msg}")
    print(f"  Emission:      {'✓' if emit_match else '✗'} - {emit_msg}")
    print(f"  Overall: {'✓' if overall_match else '✗'}")
    
    return overall_match

def run_backward_algorithm(config_file):
    """Run backward algorithm (problem 4) and return probabilities"""
    N, M, start_p, trans_p, emit_p, sequences = parse_config_file(config_file)
    
    # Create HMM
    hmm = HiddenMarkovModel(trans_p, emit_p)
    hmm.A_start = start_p
    
    results = []
    for seq in sequences:
        prob = hmm.probability_betas(seq)
        results.append(prob)
    
    return results

def test_backward_algorithm(config_file, hmm_executable="./hmm"):
    """Test backward algorithm (problem 4)"""
    print(f"Testing Backward Algorithm with {config_file}")
    
    # Run Python implementation
    python_probs = run_backward_algorithm(config_file)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "4", hmm_executable=hmm_executable)
    cpp_probs = parse_cpp_forward_output(cpp_output)  # Same parsing as forward
    
    # Compare results
    match, message = compare_probabilities(python_probs, cpp_probs)
    
    print(f"  Python results: {[f'{p:.6e}' for p in python_probs]}")
    print(f"  C++ results:    {[f'{p:.6e}' for p in cpp_probs]}")
    print(f"  Match: {'✓' if match else '✗'} - {message}")
    
    return match

def run_comprehensive_test(test_dir="test_configs", hmm_executable="./hmm"):
    """Run comprehensive test suite"""
    print("Running Comprehensive HMM Test Suite")
    print("=" * 50)
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Run generate_test_configs.py first.")
        return False
    
    config_files = [f for f in os.listdir(test_dir) if f.endswith('.cfg')]
    
    if not config_files:
        print(f"No configuration files found in {test_dir}")
        return False
    
    total_tests = 0
    passed_tests = 0
    
    for config_file in sorted(config_files):
        config_path = os.path.join(test_dir, config_file)
        print(f"\n--- Testing {config_file} ---")
        
        try:
            # Test Forward Algorithm
            if test_forward_algorithm(config_path, hmm_executable):
                passed_tests += 1
            total_tests += 1
            
            print()
            
            # Test Viterbi Algorithm
            if test_viterbi_algorithm(config_path, hmm_executable):
                passed_tests += 1
            total_tests += 1
            
            print()
            
            # Test Backward Algorithm
            if test_backward_algorithm(config_path, hmm_executable):
                passed_tests += 1
            total_tests += 1
            
            print()
            
            # Test Baum-Welch Algorithm (with fewer iterations for speed)
            if test_baum_welch_algorithm(config_path, iterations=5, hmm_executable=hmm_executable):
                passed_tests += 1
            total_tests += 1
            
        except Exception as e:
            print(f"  Error testing {config_file}: {e}")
            total_tests += 4  # Each config file has 4 tests now
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All tests passed!")
    else:
        print(f"{total_tests - passed_tests} tests failed.")
    
    return passed_tests == total_tests

def main():
    parser = argparse.ArgumentParser(description='Test HMM implementation')
    parser.add_argument('--config', '-c', help='Single configuration file to test')
    parser.add_argument('--problem', '-p', choices=['1', '2', '3', '4'], help='Specific problem to test')
    parser.add_argument('--hmm-exe', default='./hmm', help='Path to C++ HMM executable')
    parser.add_argument('--test-dir', default='test_configs', help='Directory containing test configurations')
    parser.add_argument('--iterations', '-n', type=int, default=10, help='Iterations for Baum-Welch')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    
    args = parser.parse_args()
    
    if args.comprehensive:
        success = run_comprehensive_test(args.test_dir, args.hmm_exe)
        sys.exit(0 if success else 1)
    
    if not args.config:
        print("Please specify --config or use --comprehensive")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.hmm_exe):
        print(f"HMM executable not found: {args.hmm_exe}")
        print("Make sure to compile your C++ implementation first")
        sys.exit(1)
    
    success = True
    
    if args.problem is None or args.problem == '1':
        success &= test_forward_algorithm(args.config, args.hmm_exe)
    
    if args.problem is None or args.problem == '2':
        success &= test_viterbi_algorithm(args.config, args.hmm_exe)
    
    if args.problem is None or args.problem == '3':
        success &= test_baum_welch_algorithm(args.config, args.iterations, args.hmm_exe)
    
    if args.problem is None or args.problem == '4':
        success &= test_backward_algorithm(args.config, args.hmm_exe)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 