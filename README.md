# CUDA HMM Project
A C++ project using CMake for building.

## Project Structure

```
├── CMakeLists.txt
├── include/       
├── src/         
└── README.md
```

## Building the Project

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
After building, you can run the program from the build directory:
```bash
./bin/cuda-hmm
``` 