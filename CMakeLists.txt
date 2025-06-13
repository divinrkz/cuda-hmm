cmake_minimum_required(VERSION 3.24)
project(cuda_hmm LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# Compiler settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Compile only for the architecture present on the build machine.  
# For distributing binaries you may list multiple values, e.g. "75;86".
set(CMAKE_CUDA_ARCHITECTURES native)

# -----------------------------------------------------------------------------
# Dependencies – we only need the CUDA runtime/library
# -----------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# -----------------------------------------------------------------------------
# Source group definitions
# -----------------------------------------------------------------------------
set(HMM_CPU_SOURCES
    src/hmm/hmm_impl.cpp
)

set(HMM_CUDA_SOURCES
    src/hmm/hmm_gpu.cu
)

set(HMM_PUBLIC_HEADERS
    include/hmm.hpp
    include/hmm_gpu.cuh
    include/data_loader.hpp
)

# -----------------------------------------------------------------------------
# Libraries
# -----------------------------------------------------------------------------
# CPU implementation library
add_library(hmm_cpu STATIC ${HMM_CPU_SOURCES} ${HMM_PUBLIC_HEADERS})

target_include_directories(hmm_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_property(TARGET hmm_cpu PROPERTY CXX_STANDARD 17)

# GPU implementation library
add_library(hmm_cuda STATIC ${HMM_CUDA_SOURCES} ${HMM_PUBLIC_HEADERS})

target_include_directories(hmm_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Enable separate compilation and device symbol resolution for the CUDA code
set_property(TARGET hmm_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET hmm_cuda PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET hmm_cuda PROPERTY CUDA_STANDARD 17)
set_property(TARGET hmm_cuda PROPERTY CXX_STANDARD 17)

# Link required CUDA libraries
set(CUDA_LIBS
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)

target_link_libraries(hmm_cuda PUBLIC ${CUDA_LIBS})

# Allow device code to use certain host constexpr functions (numeric_limits, etc.)
target_compile_options(hmm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# -----------------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------------
add_executable(hmm_runner src/main.cpp)

# Link both implementations (only one will be used at runtime depending on flag)
target_link_libraries(hmm_runner PRIVATE hmm_cpu hmm_cuda ${CUDA_LIBS})

target_include_directories(hmm_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_property(TARGET hmm_runner PROPERTY CXX_STANDARD 17)
set_property(TARGET hmm_runner PROPERTY CUDA_STANDARD 17)

# -----------------------------------------------------------------------------
# Testing infrastructure (optional)
# -----------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
    enable_testing()

    # Example: Python correctness tests (needs Python3 & test script present)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME Correctness_CPU
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_correctness_tests.py --impl cpu --exe $<TARGET_FILE:hmm_runner>
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        add_test(
            NAME Correctness_GPU
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_correctness_tests.py --impl gpu --exe $<TARGET_FILE:hmm_runner>
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# Installation targets
# -----------------------------------------------------------------------------
install(TARGETS hmm_runner
        RUNTIME DESTINATION bin)

# -----------------------------------------------------------------------------
# Status output
# -----------------------------------------------------------------------------
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Executable will be installed to bin/hmm_runner")
