cmake_minimum_required(VERSION 3.10)

# Project name and language
project(cuda-hmm VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable testing
enable_testing()

# Add source files
add_executable(${PROJECT_NAME} src/main.cpp)

# Add test executable
add_executable(test_hmm tests/test_hmm.cpp src/hmm/hmm_impl.cpp)
target_include_directories(test_hmm PRIVATE include src)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add test
add_test(NAME test_hmm COMMAND test_hmm) 