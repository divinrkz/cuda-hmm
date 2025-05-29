#include "hmm.hpp"
#include "data_loader.hpp"
#include <iostream>
#include <cassert>
#include <iomanip>

class HMMTester {
public:
    static void runAllTests() {
        std::cout << "=== Running HMM Tests ===" << std::endl;
        
        // testDataLoading();
        testAllSequenceFiles();
        
        std::cout << "\n=== All Tests Completed! ===" << std::endl;
    }

    
    static void testAllSequenceFiles() {
        std::cout << "\n--- Testing All Sequence Files ---" << std::endl;
        
        for (int n = 0; n < 6; n++) {
            testSequenceFile(n);
        }
    }
    
    static void testSequenceFile(int n) {
        std::string path = "../tests/data/sequence_data" + std::to_string(n) + ".txt";
        
        HMMData data;
        
        try {
            data = HMMDataLoader::loadFromFile(path);
        } catch (const std::exception&) {
            std::cout << "Could not load sequence_data" << n << ".txt from any location" << std::endl;
            return;
        }
        
        std::cout << "\nFile #" << n << ":" << std::endl;
        std::cout << std::left << std::setw(30) << "Emission Sequence" 
                    << std::setw(30) << "Max Probability State Sequence" << std::endl;
        std::cout << std::string(70, '#') << std::endl;
        
        float* trans_p = HMMDataLoader::convert2DTo1D(data.A);
        float* emit_p = HMMDataLoader::convert2DTo1D(data.O);
        float* start_p = HMMDataLoader::createUniformStartProbs(data.N);
        
        IHMM hmm(data.N, data.M);
        
        // Test each sequence
        for (const auto& sequence : data.sequences) {
            // Convert sequence to format needed by viterbi
            float* obs = HMMDataLoader::convertSequenceToFloat(sequence);
            int* states = new int[sequence.length()]; // Input parameter (not modified)
            
            // Run Viterbi
            std::string result = hmm.viterbi(obs, states, start_p, trans_p, emit_p, 
                                            sequence.length(), data.N, data.M);
            
            // Print results in same format as Python
            std::cout << std::left << std::setw(30) << sequence 
                        << std::setw(30) << result << std::endl;
            
            // Basic validation
            assert(!result.empty());
            
            // Clean up
            delete[] obs;
            delete[] states;
        }
        
        std::cout << std::endl;
        
        // Clean up matrices
        delete[] trans_p;
        delete[] emit_p;
        delete[] start_p;
        
        std::cout << "✓ File " << n << " test passed" << std::endl;
    
    }
};

int main() {
    std::cout << "HMM Real Data Tester" << std::endl;
    std::cout << "====================" << std::endl;
    
    HMMTester::runAllTests();
    
    return 0;
}