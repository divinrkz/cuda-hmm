#include "hmm.hpp"
#include "hmm_cuda.hpp"
#include "data_loader.hpp"
#include <iostream>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

class HMMGPUTester {
private:
    static void printGPUInfo() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        
        if (deviceCount == 0) {
            std::cout << "No CUDA devices found!" << std::endl;
            return;
        }
        
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        
        std::cout << "GPU Device: " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Global Memory: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << std::string(50, '=') << std::endl;
    }

public:
    static void runAllTests() {
        std::cout << "\n=== GPU HMM Performance Tests ===" << std::endl;
        printGPUInfo();
        
        testAllSequenceFiles();
        testPerformanceComparison();
        testLargeSequences();
        
        std::cout << "\n=== All GPU Tests Completed! ===" << std::endl;
    }

    static void testAllSequenceFiles() {
        std::cout << "\n--- Testing All Sequence Files (GPU) ---" << std::endl;
        for (int n = 0; n < 6; n++) {
            testViterbiSequenceFile(n);
        }
    }
    
    static void testViterbiSequenceFile(int n) {
        std::string path = "../tests/data/sequence_data" + std::to_string(n) + ".txt";
        HMMData data;
        
        try {
            data = HMMDataLoader::loadFromFile(path);
        } catch (const std::exception&) {
            std::cout << "Could not load sequence_data" << n << ".txt from any location" << std::endl;
            return;
        }
        
        std::cout << "\nFile #" << n << " (States: " << data.N << ", Observations: " << data.M << "):" << std::endl;
        std::cout << std::left << std::setw(30) << "Emission Sequence" 
                  << std::setw(30) << "Max Probability State Sequence" 
                  << std::setw(15) << "\t\t\tGPU Time (ms)" << std::endl;
        std::cout << std::string(85, '#') << std::endl;
        
        float* trans_p = HMMDataLoader::convert2DTo1D(data.A);
        float* emit_p = HMMDataLoader::convert2DTo1D(data.O);
        float* start_p = HMMDataLoader::createUniformStartProbs(data.N);
        
        HMMCuda hmm_gpu(data.N, data.M);

        for (const auto& sequence : data.sequences) {
            float* obs = HMMDataLoader::convertSequenceToFloat(sequence);
            int* states = new int[sequence.length()]; 
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::string result = hmm_gpu.viterbi(obs, states, start_p, trans_p, emit_p, 
                                               sequence.length(), data.N, data.M);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double gpu_time_ms = duration.count() / 1000.0;
            
    
            std::cout << std::left << std::setw(30) << sequence 
                      << std::setw(30) << result 
                      << std::setw(15) << std::fixed << std::setprecision(3) << gpu_time_ms << std::endl;
            
            assert(!result.empty());
            
            delete[] obs;
            delete[] states;
        }
        
        std::cout << std::endl;
    
        delete[] trans_p;
        delete[] emit_p;
        delete[] start_p;
        
        std::cout << "GPU File " << n << " test passed" << std::endl;
    }
    
    static void testPerformanceComparison() {
        std::cout << "\n--- GPU vs CPU Performance Comparison ---" << std::endl;
        
        std::vector<std::pair<int, int>> test_configs = {
            {10, 5},    // Small: 10 states, 5 observations
            {50, 20},   // Medium: 50 states, 20 observations  
            {60, 20},   // Optimal: 50 states, 20 observations  
            {100, 50},  // Large: 100 states, 50 observations
            {500, 100}  // Very Large: 500 states, 100 observations

        };
        
        std::vector<int> sequence_lengths = {100, 500, 1000, 5000};
        
        std::cout << std::left << std::setw(15) << "States" 
                  << std::setw(15) << "Observations" 
                  << std::setw(15) << "Seq Length"
                  << std::setw(15) << "CPU Time (ms)"
                  << std::setw(15) << "GPU Time (ms)" 
                  << std::setw(15) << "Speedup" << std::endl;

        
        for (auto config : test_configs) {
            int N = config.first;   
            int M = config.second;  
            
            for (int T : sequence_lengths) {
         
                float* obs = generateRandomSequence(T, M);
                int* states = new int[T];
                float* start_p = generateUniformProbs(N);
                float* trans_p = generateRandomProbs(N * N);
                float* emit_p = generateRandomProbs(N * M);
                
                IHMM hmm_cpu(N, M);
                auto cpu_start = std::chrono::high_resolution_clock::now();
                std::string cpu_result = hmm_cpu.viterbi(obs, states, start_p, trans_p, emit_p, T, N, M);
                auto cpu_end = std::chrono::high_resolution_clock::now();
                double cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;
                
                HMMCuda hmm_gpu(N, M);
                auto gpu_start = std::chrono::high_resolution_clock::now();
                std::string gpu_result = hmm_gpu.viterbi(obs, states, start_p, trans_p, emit_p, T, N, M);
                auto gpu_end = std::chrono::high_resolution_clock::now();
                double gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;
                
                double speedup = cpu_time / gpu_time;
                
                std::cout << std::left << std::setw(15) << N 
                          << std::setw(15) << M 
                          << std::setw(15) << T
                          << std::setw(15) << std::fixed << std::setprecision(2) << cpu_time
                          << std::setw(15) << std::fixed << std::setprecision(2) << gpu_time
                          << std::setw(15) << std::fixed << std::setprecision(1) << speedup << std::endl;
                
                // Cleanup
                delete[] obs;
                delete[] states;
                delete[] start_p;
                delete[] trans_p;
                delete[] emit_p;
            }
        }
    }
    
    static void testLargeSequences() {
        std::cout << "\n--- Testing Large Sequences (GPU Only) ---" << std::endl;
        
        std::vector<std::tuple<int, int, int>> large_configs = {
            {100, 50, 10000},   // 100 states, 50 obs, 10k sequence
            {200, 100, 5000},   // 200 states, 100 obs, 5k sequence
            {500, 200, 2000},   // 500 states, 200 obs, 2k sequence
            {1000, 500, 1000}   // 1000 states, 500 obs, 1k sequence
        };
        
        std::cout << std::left << std::setw(15) << "States" 
                  << std::setw(15) << "Observations" 
                  << std::setw(15) << "Seq Length"
                  << std::setw(15) << "GPU Time (ms)"
                  << std::setw(20) << "Memory Usage (MB)" << std::endl;
        std::cout << std::string(95, '-') << std::endl;
        
        for (auto config : large_configs) {
            int N = std::get<0>(config);
            int M = std::get<1>(config);
            int T = std::get<2>(config);
            
            // Calculate approximate memory usage
            size_t memory_mb = (T * N * sizeof(float) * 2 +  // alpha, beta arrays
                               T * N * sizeof(int) +          // path array
                               N * N * sizeof(float) +        // transition matrix
                               N * M * sizeof(float)) / (1024 * 1024);  // emission matrix
            
            try {
                float* obs = generateRandomSequence(T, M);
                int* states = new int[T];
                float* start_p = generateUniformProbs(N);
                float* trans_p = generateRandomProbs(N * N);
                float* emit_p = generateRandomProbs(N * M);
                
                HMMCuda hmm_gpu(N, M);
                auto start_time = std::chrono::high_resolution_clock::now();
                std::string result = hmm_gpu.viterbi(obs, states, start_p, trans_p, emit_p, T, N, M);
                auto end_time = std::chrono::high_resolution_clock::now();
                
                double gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
                
                std::cout << std::left << std::setw(15) << N 
                          << std::setw(15) << M 
                          << std::setw(15) << T
                          << std::setw(15) << std::fixed << std::setprecision(2) << gpu_time
                          << std::setw(20) << memory_mb << std::endl;
                
                assert(!result.empty());
                
                delete[] obs;
                delete[] states;
                delete[] start_p;
                delete[] trans_p;
                delete[] emit_p;
                
            } catch (const std::exception& e) {
                std::cout << "Failed for config (" << N << "," << M << "," << T << "): " << e.what() << std::endl;
            }
        }
    }

private:
    static float* generateRandomSequence(int length, int max_obs) {
        float* seq = new float[length];
        for (int i = 0; i < length; i++) {
            seq[i] = static_cast<float>(rand() % max_obs);
        }
        return seq;
    }
    
    static float* generateUniformProbs(int size) {
        float* probs = new float[size];
        float uniform_prob = 1.0f / size;
        for (int i = 0; i < size; i++) {
            probs[i] = uniform_prob;
        }
        return probs;
    }
    
    static float* generateRandomProbs(int size) {
        float* probs = new float[size];
        float sum = 0.0f;
        
        // Generate random values
        for (int i = 0; i < size; i++) {
            probs[i] = static_cast<float>(rand()) / RAND_MAX;
            sum += probs[i];
        }
        
        // Normalize to make them probabilities
        for (int i = 0; i < size; i++) {
            probs[i] /= sum;
        }
        
        return probs;
    }
};

int main() {
    std::cout << "Cuda HMM Tests" << std::endl;
    std::cout << "====================" << std::endl;
    
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA initialization failed!" << std::endl;
        return 1;
    }
    // for reproducible tests
    srand(42);
    
    HMMGPUTester::runAllTests();
    
    cudaDeviceReset();
    
    return 0;
}