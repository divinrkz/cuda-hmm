# CUDA HMM

## Installation and Usage Instructions
The project can be built and run following the instructions in the "Building the project" and "Usage" sections below.

### Project Description and Features
This project is a parallel implementation of Hidden Markov Model (HMM) algorithms. Find our parallelization strategy [here.](https://docs.google.com/document/d/1zd5hZewnfNOuB6ldeuOxLj-NUDal9cU6siH8P-q5ZBU/edit?usp=sharing)


### Expected Results and Performance Analysis
The GPU implementation shows significant speedup compared to the CPU version, especially for:
- Large observation sequences
- Complex HMM models with many states.



### Potential Improvements
Find other potential optimazations [here.](https://docs.google.com/document/d/1zd5hZewnfNOuB6ldeuOxLj-NUDal9cU6siH8P-q5ZBU/edit?usp=sharing)

----------------------
### Building the project
#### Prerequisites
- CMake (version 3.10 or higher)
- C++ compiler with C++17 support
- Make 

#### Build Instructions
1. Create a build directory:
```bash
mkdir build
cd build
```
2. Generate build files and run make"
```bash
cmake ..
make -j
```
The executable `hmm_runner` will be created in the `build/` directory.

#### Usage
Run HMM algorithms on configuration files:

```bash
# Forward algorithm
./build/hmm_runner -c test_configs/{filename}.cfg -p1 --impl {cpu,gpu,all}

# Viterbi algorithm  
./build/hmm_runner -c test_configs/{filename}.cfg -p2 --impl {cpu,gpu,all}

# Baum-Welch training
./build/hmm_runner -c test_configs/{filename}.cfg -p3 -n 100  --impl {cpu,gpu,all}

# Backward algorithm
./build/hmm_runner -c test_configs/{filename}.cfg -p4  --impl {cpu,gpu,all}
```
The [filename] should be replace with any file in the `test_configs` folder.

#### HMM Correctness Tests
Test implementation against Python reference:

```bash
# Test all algorithms on all config files
python3 test_hmm.py --hmm-exe build/hmm_runner --impl {cpu,gpu,all}
```

#### Benchmark tests:
```bash
# Using CTest (recommended)
cd build

# Using test executable directly
./benchmark_test 
```

#### Contributors
- Mugisha AbdulKarim (@abdulkarim-mugisha)
- Divin Irakiza (@divinrkz)