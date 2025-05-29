#include "hmm.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

class HMMTester {
public:
    static void runAllTests() {
        std::cout << "=== Running HMM Tests ===" << std::endl;
        
        testConstructorDestructor();
        testSimpleViterbi();
        testSingleObservation();
        testLongerSequence();
        
        std::cout << "\n=== All Tests Passed! ===" << std::endl;
    }

private:
    static void testConstructorDestructor() {
        std::cout << "\n--- Testing Constructor/Destructor ---" << std::endl;
        
        IHMM* hmm = new IHMM(3, 4);
        std::cout << "✓ Constructor works" << std::endl;
        
        delete hmm;
        std::cout << "✓ Destructor works" << std::endl;
    }

    static void testSimpleViterbi() {
        std::cout << "\n--- Testing Simple Viterbi (2 states, 2 observations) ---" << std::endl;
        
        int N = 2, M = 2, T = 3;
        float start_p[] = {0.6f, 0.4f};
        
        float trans_p[] = {0.7f, 0.3f,  
                          0.4f, 0.6f};  
        
        float emit_p[] = {0.9f, 0.1f,  
                         0.2f, 0.8f};  
        
        float obs[] = {0.0f, 1.0f, 0.0f};
        int states[3] = {0}; 
        
        IHMM hmm(N, M);
        std::string result = hmm.viterbi(obs, states, start_p, trans_p, emit_p, T, N, M);
        
        std::cout << "Computed path: " << result << std::endl;
        

        assert(result.find("0") != std::string::npos); 
        assert(result.find("1") != std::string::npos); 
        assert(result.find("->") != std::string::npos);
        
        std::cout << "✓ Simple Viterbi test passed" << std::endl;
    }


    static void testSingleObservation() {
        std::cout << "\n--- Testing Single Observation ---" << std::endl;
        
        int N = 3, M = 2, T = 1;
        
        float start_p[] = {0.1f, 0.6f, 0.3f};
        float trans_p[] = {1.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 1.0f};  
        
        float emit_p[] = {0.9f, 0.1f, 
                         0.3f, 0.7f, 
                         0.5f, 0.5f};
        
        float obs[] = {0.0f};
        int states[1] = {0}; // Input array
        
        IHMM hmm(N, M);
        std::string result = hmm.viterbi(obs, states, start_p, trans_p, emit_p, T, N, M);
        
        assert(result == "1");
        
        std::cout << "✓ Single observation test passed" << std::endl;
    }

    static void testLongerSequence() {
        std::cout << "\n--- Testing Longer Sequence (10 steps) ---" << std::endl;
        
        int N = 2, M = 2, T = 10;
        
        float start_p[] = {0.5f, 0.5f};
        float trans_p[] = {0.8f, 0.2f,
                          0.3f, 0.7f};
        float emit_p[] = {0.9f, 0.1f,
                         0.1f, 0.9f};
        
        float obs[] = {0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        int states[10] = {0};
        
        IHMM hmm(N, M);
        std::string result = hmm.viterbi(obs, states, start_p, trans_p, emit_p, T, N, M);
        
        std::cout << "Long sequence results:" << std::endl;
        std::cout << "Obs:    ";
        for(int i = 0; i < T; i++) std::cout << static_cast<int>(obs[i]) << " ";
        std::cout << std::endl;
        std::cout << "States: " << result << std::endl;
        
        assert(!result.empty());
        assert(result.find("->") != std::string::npos);
        
        std::cout << "✓ Longer sequence test passed" << std::endl;
    }

};

void printMatrices(float* start_p, float* trans_p, float* emit_p, int N, int M) {
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "Initial probabilities: ";
    for(int i = 0; i < N; i++) {
        std::cout << start_p[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Transition matrix:" << std::endl;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            std::cout << trans_p[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Emission matrix:" << std::endl;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < M; j++) {
            std::cout << emit_p[i * M + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    HMMTester::runAllTests();

    return 0;
}