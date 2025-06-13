cmake_minimum_required(VERSION 3.24)  # for CUDA architecture native
project(cuda-hmm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES native)

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories(${PROJECT_NAME} PRIVATE include)
include_directories(${PROJECT_NAME} PRIVATE src)


enable_testing()

set(HMM_CPU_SOURCES
    src/hmm/hmm_impl.cpp
    include/hmm.hpp
    include/data_loader.hpp
)

set(HMM_CUDA_SOURCES
    src/hmm/hmm_gpu.cu
    include/hmm.hpp
    include/hmm_gpu.cuh
    include/data_loader.hpp
)


add_library(hmm_cpu_lib STATIC ${HMM_CPU_SOURCES})
target_include_directories(hmm_cpu_lib PUBLIC include)

add_library(hmm_cuda_lib STATIC ${HMM_CUDA_SOURCES})
target_include_directories(hmm_cuda_lib PUBLIC include)
set_property(TARGET hmm_cuda_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET hmm_cuda_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)


add_executable(test_hmm_cpu tests/test_hmm.cpp)
target_link_libraries(test_hmm_cpu hmm_cpu_lib)


add_executable(test_hmm_gpu tests/test_hmm_gpu.cpp)  
target_link_libraries(test_hmm_gpu hmm_cpu_lib hmm_cuda_lib)


set(CUDA_LINK_LIBRARIES
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

target_link_libraries(test_hmm_gpu ${CUDA_LINK_LIBRARIES})
set_property(TARGET test_hmm_gpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET test_hmm_gpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)


add_executable(hmm src/main.cpp)
target_link_libraries(hmm hmm_cpu_lib hmm_cuda_lib ${CUDA_LINK_LIBRARIES})

add_executable(hmm_gpu src/main_gpu.cpp)
target_link_libraries(hmm_gpu hmm_cpu_lib hmm_cuda_lib ${CUDA_LINK_LIBRARIES})
set_property(TARGET hmm_gpu PROPERTY CXX_STANDARD 17)
set_property(TARGET hmm_gpu PROPERTY CUDA_STANDARD 17)


target_compile_options(hmm_cuda_lib PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--keep;-src-in-ptx;--generate-line-info;--extended-lambda>
)

target_compile_options(test_hmm_gpu PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--keep;-src-in-ptx;--generate-line-info;--extended-lambda>
)

target_compile_options(hmm PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--keep;-src-in-ptx;--generate-line-info;--extended-lambda>
)


set_property(TARGET hmm_cpu_lib PROPERTY CXX_STANDARD 17)
set_property(TARGET hmm_cuda_lib PROPERTY CXX_STANDARD 17)
set_property(TARGET test_hmm_cpu PROPERTY CXX_STANDARD 17)
set_property(TARGET test_hmm_gpu PROPERTY CXX_STANDARD 17)
set_property(TARGET hmm PROPERTY CXX_STANDARD 17)

set_property(TARGET hmm_cuda_lib PROPERTY CUDA_STANDARD 17)
set_property(TARGET test_hmm_gpu PROPERTY CUDA_STANDARD 17)
set_property(TARGET hmm PROPERTY CUDA_STANDARD 17)

# Add tests
add_test(NAME test_hmm_cpu_suite COMMAND test_hmm_cpu)
add_test(NAME test_hmm_gpu_suite COMMAND test_hmm_gpu)

set_tests_properties(test_hmm_cpu_suite PROPERTIES 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(test_hmm_gpu_suite PROPERTIES 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


install(TARGETS hmm hmm_gpu test_hmm_cpu test_hmm_gpu
    RUNTIME DESTINATION bin
)


message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Include Dirs: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")


add_custom_target(run_gpu_tests
    COMMAND test_hmm_gpu
    DEPENDS test_hmm_gpu
    COMMENT "Running GPU HMM tests..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)


add_custom_target(run_all_tests
    COMMAND test_hmm_cpu && test_hmm_gpu
    DEPENDS test_hmm_cpu test_hmm_gpu
    COMMENT "Running all HMM tests..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)