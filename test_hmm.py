# File: run_correctness_tests.py
import os
import sys
import subprocess
import argparse
import numpy as np

# Import the functions from your provided Python reference implementation.
# Make sure your file is named 'hmm_reference.py' in the same directory.
try:
    from hmm_reference import (
        parse_config_file,
        run_viterbi_algorithm,
        run_baum_welch_algorithm
    )
except ImportError:
    print("ERROR: Could not find 'hmm_reference.py'. Please ensure it's in the same directory.", file=sys.stderr)
    sys.exit(1)

# --- C++ Runner and Parsers (Helper functions to interact with the C++ executable) ---

def run_cpp_executable(config_file, problem, executable_path, iterations=100, impl_type='cpu'):
    """
    Generic function to run the compiled C++ HMM executable.
    It constructs the command line arguments and captures the output.
    """
    if not os.path.exists(executable_path):
        raise FileNotFoundError(f"C++ HMM executable not found: {executable_path}")

    cmd = [executable_path, "--impl", impl_type, "-c", config_file, f"-p{problem}"]
    if problem == 3:
        cmd.extend(["-n", str(iterations)])

    try:
        # Run the subprocess, capture output, and check for errors.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=90)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"C++ executable failed (exit code {e.returncode}) on '{config_file}':\n--- STDERR ---\n{e.stderr}\n--------------")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"C++ executable timed out on '{config_file}'")

def parse_viterbi_output(output_str):
    """Parses the multi-line output of the Viterbi algorithm into a list of strings."""
    return [line.strip().replace(' ', '') for line in output_str.split('\n') if line.strip()]

def parse_baum_welch_output(output_str):
    """Parses the matrix-formatted output of the Baum-Welch algorithm."""
    lines = [line.strip() for line in output_str.split('\n') if line.strip() and not line.startswith('#')]
    if not lines:
        return None, None, None

    # Determine N from the length of the first line (initial probabilities)
    N = len(lines[0].split())
    
    start_p = np.array([float(x) for x in lines[0].split()], dtype=np.float32)
    trans_p = np.array([[float(x) for x in line.split()] for line in lines[1 : 1 + N]], dtype=np.float32)
    emit_p = np.array([[float(x) for x in line.split()] for line in lines[1 + N:]], dtype=np.float32)
    
    return start_p, trans_p, emit_p

# --- Comparison Logic (Asserts for correctness) ---

def compare_sequences(python_seqs, cpp_seqs):
    """Asserts that two lists of Viterbi state sequences are identical."""
    assert len(python_seqs) == len(cpp_seqs), \
        f"Sequence count mismatch: Python has {len(python_seqs)}, C++ has {len(cpp_seqs)}"
    for i, (p_seq, c_seq) in enumerate(zip(python_seqs, cpp_seqs)):
        assert p_seq == c_seq, \
            f"Sequence mismatch at index {i}:\n  Python Ref: '{p_seq}'\n  C++ Output: '{c_seq}'"

def compare_matrices(p_mat, c_mat, name, tolerance=1e-4):
    """Asserts that two numpy arrays are close within a given tolerance."""
    assert p_mat is not None and c_mat is not None, f"One of the '{name}' matrices is None"
    assert p_mat.shape == c_mat.shape, \
        f"{name} shape mismatch: Python Ref is {p_mat.shape}, C++ Output is {c_mat.shape}"
    assert np.allclose(p_mat, c_mat, atol=tolerance), \
        f"{name} mismatch beyond tolerance {tolerance}.\nPY: {p_mat}\nC++: {c_mat}"

# --- Test Functions (Orchestrate running Python and C++ and comparing) ---

def test_viterbi(config_path, executable_path, impl_type):
    """Runs a correctness test for the Viterbi algorithm."""
    print(f"  [Viterbi]... ", end="", flush=True)
    
    # 1. Run Python reference implementation to get the ground truth
    py_seqs = run_viterbi_algorithm(config_path)
    
    # 2. Run the C++ implementation
    cpp_output = run_cpp_executable(config_path, 2, executable_path, impl_type=impl_type)
    cpp_seqs = parse_viterbi_output(cpp_output)
    
    # 3. Compare the results
    compare_sequences(py_seqs, cpp_seqs)
    print("✓ PASSED")

def test_baum_welch(config_path, executable_path, impl_type, iterations=10):
    """Runs a correctness test for the Baum-Welch algorithm."""
    print(f"  [Baum-Welch ({iterations} iters)]... ", end="", flush=True)
    
    # 1. Run Python reference implementation
    py_start, py_trans, py_emit = run_baum_welch_algorithm(config_path, n_iters=iterations)
    
    # 2. Run the C++ implementation
    cpp_output = run_cpp_executable(config_path, 3, executable_path, iterations=iterations, impl_type=impl_type)
    cpp_start, cpp_trans, cpp_emit = parse_baum_welch_output(cpp_output)
    
    # 3. Compare the results
    compare_matrices(np.array(py_start, dtype=np.float32), cpp_start, "Initial Probs")
    compare_matrices(np.array(py_trans, dtype=np.float32), cpp_trans, "Transition Matrix")
    compare_matrices(np.array(py_emit, dtype=np.float32), cpp_emit, "Emission Matrix")
    print("✓ PASSED")

def run_test_suite_for_impl(test_dir, executable_path, impl_type):
    """
    Finds all .cfg files in a directory and runs the test suite for a specific
    implementation (cpu or gpu).
    """
    print("-" * 60)
    print(f"Running Correctness Tests for: {impl_type.upper()} implementation")
    print(f"(Using executable: {executable_path})")
    print("-" * 60)
    
    if not os.path.exists(test_dir):
        print(f"ERROR: Test directory '{test_dir}' not found.", file=sys.stderr)
        return False
        
    config_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.cfg')])
    if not config_files:
        print(f"ERROR: No .cfg files found in '{test_dir}'", file=sys.stderr)
        return False

    passed_configs = 0
    for config_file in config_files:
        config_path = os.path.join(test_dir, config_file)
        print(f"Testing with '{config_file}':")
        try:
            # Run all tests for this config file
            test_viterbi(config_path, executable_path, impl_type)
            test_baum_welch(config_path, executable_path, impl_type, iterations=5)
            passed_configs += 1
            print("  -> CONFIG PASSED\n")
        except (AssertionError, RuntimeError, FileNotFoundError) as e:
            print(f"✗ FAILED: {e}\n")
        except Exception as e:
            print(f"✗ UNEXPECTED ERROR: {e}\n")

    print("-" * 60)
    if passed_configs == len(config_files):
        print(f"SUCCESS: All {passed_configs}/{len(config_files)} config files passed for {impl_type.upper()}.")
        return True
    else:
        print(f"FAILURE: {len(config_files) - passed_configs} config files failed for {impl_type.upper()}.")
        return False

# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="HMM Correctness Test Suite")
    parser.add_argument('--exe', default='./hmm_runner', help='Path to the compiled C++ HMM runner executable.')
    parser.add_argument('--test-dir', default='test_configs', help='Directory containing .cfg test files.')
    parser.add_argument('--impl', choices=['cpu', 'gpu', 'all'], default='all', help='Which implementation(s) to test.')
    
    args = parser.parse_args()

    overall_success = True
    
    # Test CPU implementation if requested
    if args.impl in ['cpu', 'all']:
        if not run_test_suite_for_impl(args.test_dir, args.exe, 'cpu'):
            overall_success = False
            
    # Test GPU implementation if requested
    if args.impl in ['gpu', 'all']:
        if not run_test_suite_for_impl(args.test_dir, args.exe, 'gpu'):
            overall_success = False
            
    if overall_success:
        print("\n🎉 All specified tests passed successfully! 🎉")
        sys.exit(0)
    else:
        print("\n❌ One or more tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()