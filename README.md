# CUDA HMM Project
Here is a guide to run the project.

## Project Structure

```
├── CMakeLists.txt
├── include/       
├── src/   
├────── hmm/         
├──────── hmm_impl.cpp
├────── main.cpp    
├── tests/   
├────── test_hmm.cpp         
└── README.md
```

## Building the project
### Prerequisites

- CMake (version 3.10 or higher)
- C++ compiler with C++17 support
- Make 

### Build Instructions

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Generate build files:
```bash
cmake ..
```

3. Build the project:
```bash
cmake --build .
```

The executable will be created in the `build/bin` directory.

## Usage

Run HMM algorithms on configuration files:

```bash
# Forward algorithm
./build/cuda-hmm -c config_file.cfg -p1

# Viterbi algorithm  
./build/cuda-hmm -c config_file.cfg -p2

# Baum-Welch training
./build/cuda-hmm -c config_file.cfg -p3 -n 100

# Backward algorithm
./build/cuda-hmm -c config_file.cfg -p4
```

## Testing

Test implementation against Python reference:

```bash
# Test all algorithms on all config files
python3 test_hmm.py --comprehensive --hmm-exe ./build/cuda-hmm

# Test specific config file
python3 test_hmm.py --config test_configs/coin_flip.cfg --hmm-exe ./build/cuda-hmm

# Test specific algorithm (1=forward, 2=viterbi, 3=baum-welch, 4=backward)
python3 test_hmm.py --config test_configs/weather.cfg --problem 2 --hmm-exe ./build/cuda-hmm
```

Run unit tests:

```bash
# Using CTest (recommended)
cd build && ctest

# Using test executable directly
./build/test_hmm
```

## Contributors
- Mugisha AbdulKarim (@abdulkarim-mugisha)
- Divin Irakiza (@divinrkz)