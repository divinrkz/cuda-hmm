# REPLACE ENTIRE FILE
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
from hmm_reference import run_forward_algorithm, run_viterbi_algorithm, run_baum_welch_algorithm, HiddenMarkovModel, parse_config_file

def run_cpp_hmm(config_file, problem, impl_type='cpu', iterations=100, hmm_executable="./hmm_runner"):
    """
    Run the C++ HMM implementation
    """
    if not os.path.exists(hmm_executable):
        raise FileNotFoundError(f"C++ HMM executable not found: {hmm_executable}")
    
    # Use the --impl flag for the new unified runner
    cmd = [hmm_executable, "--impl", impl_type, "-c", config_file, f"-p{problem}"]
    if problem == "3":
        cmd.extend(["-n", str(iterations)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"C++ HMM failed for --impl {impl_type} (exit code {e.returncode}):\n{e.stderr}")
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

def test_forward_algorithm(config_file, impl_type, hmm_executable="./hmm_runner"):
    """Test forward algorithm (problem 1)"""
    print(f"Testing Forward Algorithm with {config_file} on {impl_type.upper()}")
    
    # Run Python implementation
    python_probs = run_forward_algorithm(config_file)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "1", impl_type=impl_type, hmm_executable=hmm_executable)
    cpp_probs = parse_cpp_forward_output(cpp_output)
    
    # Compare results
    match, message = compare_probabilities(python_probs, cpp_probs)
    
    print(f"  Python results: {[f'{p:.6e}' for p in python_probs]}")
    print(f"  C++ results:    {[f'{p:.6e}' for p in cpp_probs]}")
    print(f"  Match: {'✓' if match else '✗'} - {message}")
    
    return match

def test_viterbi_algorithm(config_file, impl_type, hmm_executable="./hmm_runner"):
    """Test Viterbi algorithm (problem 2)"""
    print(f"Testing Viterbi Algorithm with {config_file} on {impl_type.upper()}")
    
    # Run Python implementation
    python_seqs = run_viterbi_algorithm(config_file)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "2", impl_type=impl_type, hmm_executable=hmm_executable)
    cpp_seqs = parse_cpp_viterbi_output(cpp_output)
    
    # Compare results
    match, message = compare_sequences(python_seqs, cpp_seqs)
    
    print(f"  Python results: {python_seqs}")
    print(f"  C++ results:    {cpp_seqs}")
    print(f"  Match: {'✓' if match else '✗'} - {message}")
    
    return match

def test_baum_welch_algorithm(config_file, impl_type, iterations=10, hmm_executable="./hmm_runner"):
    """Test Baum-Welch algorithm (problem 3)"""
    print(f"Testing Baum-Welch Algorithm with {config_file} ({iterations} iterations) on {impl_type.upper()}")
    
    # Run Python implementation
    python_start, python_trans, python_emit = run_baum_welch_algorithm(config_file, iterations)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "3", impl_type=impl_type, iterations=iterations, hmm_executable=hmm_executable)
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

def test_backward_algorithm(config_file, impl_type, hmm_executable="./hmm_runner"):
    """Test backward algorithm (problem 4)"""
    print(f"Testing Backward Algorithm with {config_file} on {impl_type.upper()}")
    
    # Run Python implementation
    python_probs = run_backward_algorithm(config_file)
    
    # Run C++ implementation
    cpp_output = run_cpp_hmm(config_file, "4", impl_type=impl_type, hmm_executable=hmm_executable)
    cpp_probs = parse_cpp_forward_output(cpp_output)  # Same parsing as forward
    
    # Compare results
    match, message = compare_probabilities(python_probs, cpp_probs)
    
    print(f"  Python results: {[f'{p:.6e}' for p in python_probs]}")
    print(f"  C++ results:    {[f'{p:.6e}' for p in cpp_probs]}")
    print(f"  Match: {'✓' if match else '✗'} - {message}")
    
    return match

def run_comprehensive_test(test_dir="test_configs", impl_type="all", hmm_executable="./hmm_runner"):
    """Run comprehensive test suite for a specific implementation."""
    print("=" * 60)
    print(f"Running Comprehensive Tests for: {impl_type.upper()}")
    print(f"(Executable: {hmm_executable})")
    print("=" * 60)

    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} not found. Run generate_test_configs.py first.")
        return False
    
    config_files = [f for f in os.listdir(test_dir) if f.endswith('.cfg')]
    
    if not config_files:
        print(f"No configuration files found in {test_dir}")
        return False
    
    # Test specified implementation
    if impl_type in ['cpu', 'gpu']:
        passed_all = True
        for config_file in sorted(config_files):
            config_path = os.path.join(test_dir, config_file)
            print(f"\n--- Testing {config_file} ---")
            
            try:
                # Test all algorithms for this config file
                fwd_ok = test_forward_algorithm(config_path, impl_type, hmm_executable)
                print()
                vit_ok = test_viterbi_algorithm(config_path, impl_type, hmm_executable)
                print()
                bwd_ok = test_backward_algorithm(config_path, impl_type, hmm_executable)
                print()
                bw_ok = test_baum_welch_algorithm(config_path, impl_type, iterations=5, hmm_executable=hmm_executable)
                
                if not (fwd_ok and vit_ok and bwd_ok and bw_ok):
                    passed_all = False

            except Exception as e:
                print(f"  ERROR testing {config_file} on {impl_type.upper()}: {e}")
                passed_all = False
        
        print("\n" + "=" * 60)
        if passed_all:
            print(f"SUCCESS: All tests passed for {impl_type.upper()}.")
        else:
            print(f"FAILURE: One or more tests failed for {impl_type.upper()}.")
        return passed_all

def main():
    parser = argparse.ArgumentParser(description='Test HMM implementation')
    parser.add_argument('--impl', choices=['cpu', 'gpu', 'all'], default='all', help='Implementation to test.')
    parser.add_argument('--hmm-exe', default='./build/hmm_runner', help='Path to C++ HMM executable.')
    parser.add_argument('--test-dir', default='test_configs', help='Directory containing test configurations.')
    
    args = parser.parse_args()

    overall_success = True
    if args.impl in ['cpu', 'all']:
        print("--- STARTING CPU TESTS ---")
        if not run_comprehensive_test(args.test_dir, 'cpu', args.hmm_exe):
            overall_success = False
    
    if args.impl in ['gpu', 'all']:
        print("\n--- STARTING GPU TESTS ---")
        if not run_comprehensive_test(args.test_dir, 'gpu', args.hmm_exe):
            overall_success = False

    print("\n" + "="*60)
    if overall_success:
        print("🎉 All specified test suites passed successfully! 🎉")
        sys.exit(0)
    else:
        print("❌ One or more test suites failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()