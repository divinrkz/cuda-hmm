#!/usr/bin/env python3
"""GPU test runner: delegates to test_hmm.py but points at ./hmm_gpu by default."""

import argparse
import sys
import test_hmm


def main():
    parser = argparse.ArgumentParser(description="Run HMM GPU implementation tests (wraps test_hmm.py)")
    parser.add_argument('--comprehensive', action='store_true', help='Run the full test suite')
    parser.add_argument('--config', '-c', help='Single configuration file to test')
    parser.add_argument('--problem', '-p', choices=['1', '2', '3', '4'], help='Problem number to test')
    parser.add_argument('--iterations', '-n', type=int, default=10, help='Iterations for Baum-Welch')
    parser.add_argument('--test-dir', default='test_configs', help='Directory containing test configs')
    parser.add_argument('--hmm-exe', default='./hmm_gpu', help='Path to GPU HMM executable')
    args = parser.parse_args()

    # Build argument list for test_hmm main
    new_args = [sys.argv[0]]
    if args.comprehensive:
        new_args.append('--comprehensive')
    if args.config:
        new_args.extend(['--config', args.config])
    if args.problem:
        new_args.extend(['--problem', args.problem])
    if args.iterations is not None:
        new_args.extend(['--iterations', str(args.iterations)])
    new_args.extend(['--test-dir', args.test_dir, '--hmm-exe', args.hmm_exe])

    # Delegate execution
    sys.argv = new_args
    test_hmm.main()


if __name__ == "__main__":
    main() 