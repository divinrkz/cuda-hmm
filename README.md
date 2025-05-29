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

## Running the Program
You can run the program from the build directory:
```bash
./bin/cuda-hmm
```

## Running Tests
To run the tests:

1. Make sure you've built the project.

2. From the build directory, you can run the tests in two ways:

   a. Using the test executable directly:
   ```bash
   ./test_hmm
   ```

   b. Using CTest (recommended):
   ```bash
   ctest
   ```

The tests will run through various HMM scenarios including (for now):
- Constructor/Destructor tests
- Simple Viterbi tests

Each test will print its progress and results. If any test fails, it will throw an assertion error. 

## Contributors
- Mugisha AbdulKarim (@abdulkarim-mugisha)
- Divin Irakiza (@divinrkz)