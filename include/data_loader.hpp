#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

struct HMMData {
    int N; // number of states 
    int M; // number of observation types 
    std::vector<std::vector<float>> A; // transition matrix (N x N)
    std::vector<std::vector<float>> O; // observation matrix (N x M)
    std::vector<std::string> sequences; // test sequences as strings
};

class HMMDataLoader {
public:
    static HMMData loadFromFile(const std::string& filename) {
        HMMData data;
        
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::string line;
        
        // Read first line: Y (states) and X (observation types)
        if (!std::getline(file, line)) {
            throw std::runtime_error("Cannot read first line");
        }
        
        std::istringstream firstLine(line);
        if (!(firstLine >> data.N >> data.M)) {
            throw std::runtime_error("Cannot parse Y and X from first line: " + line);
        }
        
        std::cout << "Loading HMM: " << data.N << " states, " << data.M << " observation types" << std::endl;
        
        // Initialize matrices
        data.A.resize(data.N, std::vector<float>(data.N));
        data.O.resize(data.N, std::vector<float>(data.M));
        
        // Read transition matrix A (N x N)
        std::cout << "Reading transition matrix..." << std::endl;
        for (int i = 0; i < data.N; i++) {
            if (!std::getline(file, line)) {
                throw std::runtime_error("Cannot read transition matrix row " + std::to_string(i));
            }
            
            std::istringstream rowStream(line);
            for (int j = 0; j < data.N; j++) {
                if (!(rowStream >> data.A[i][j])) {
                    throw std::runtime_error("Cannot read transition matrix element (" + 
                                           std::to_string(i) + "," + std::to_string(j) + ")");
                }
            }
        }
        
        // Read observation matrix O (N x M) 
        std::cout << "Reading observation matrix..." << std::endl;
        for (int i = 0; i < data.N; i++) {
            if (!std::getline(file, line)) {
                throw std::runtime_error("Cannot read observation matrix row " + std::to_string(i));
            }
            
            std::istringstream rowStream(line);
            for (int j = 0; j < data.M; j++) {
                if (!(rowStream >> data.O[i][j])) {
                    throw std::runtime_error("Cannot read observation matrix element (" + 
                                           std::to_string(i) + "," + std::to_string(j) + ")");
                }
            }
        }
        
        // Read sequences (rest of file - should be 5 sequences)
        std::cout << "Reading test sequences..." << std::endl;
        while (std::getline(file, line)) {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }
            
            // Remove any whitespace
            std::string cleanSeq;
            for (char c : line) {
                if (c >= '0' && c <= '9') {
                    cleanSeq += c;
                }
            }
            
            if (!cleanSeq.empty()) {
                data.sequences.push_back(cleanSeq);
                std::cout << "  Sequence " << data.sequences.size() << ": " << cleanSeq 
                          << " (length: " << cleanSeq.length() << ")" << std::endl;
            }
        }
        
        file.close();
        
        std::cout << "Successfully loaded HMM with " << data.sequences.size() << " test sequences" << std::endl;
        return data;
    }
    
    // Convert string sequence to float array for observations
    static float* convertSequenceToFloat(const std::string& sequence) {
        float* obs = new float[sequence.length()];
        for (size_t i = 0; i < sequence.length(); i++) {
            obs[i] = static_cast<float>(sequence[i] - '0'); // Convert char digit to int
        }
        return obs;
    }
    
    // Convert 2D vector to 1D array (for your current HMM implementation)
    static float* convert2DTo1D(const std::vector<std::vector<float>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        float* array = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                array[i * cols + j] = matrix[i][j];
            }
        }
        return array;
    }
    
    // Create uniform initial state probabilities
    static float* createUniformStartProbs(int numStates) {
        float* start_p = new float[numStates];
        float prob = 1.0f / numStates;
        for (int i = 0; i < numStates; i++) {
            start_p[i] = prob;
        }
        return start_p;
    }
    
    // Print HMM data for debugging
    static void printHMMData(const HMMData& data) {
        std::cout << "\n=== HMM Data Summary ===" << std::endl;
        std::cout << "States: " << data.N << ", Observation types: " << data.M << std::endl;
        
        std::cout << "\nTransition Matrix A (" << data.N << "x" << data.N << "):" << std::endl;
        for (int i = 0; i < data.N; i++) {
            for (int j = 0; j < data.N; j++) {
                std::cout << std::fixed << std::setprecision(3) << data.A[i][j] << "\t";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nObservation Matrix O (" << data.N << "x" << data.M << "):" << std::endl;
        for (int i = 0; i < data.N; i++) {
            for (int j = 0; j < data.M; j++) {
                std::cout << std::fixed << std::setprecision(3) << data.O[i][j] << "\t";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nTest Sequences (" << data.sequences.size() << " total):" << std::endl;
        for (size_t i = 0; i < data.sequences.size(); i++) {
            std::cout << "  " << i+1 << ": " << data.sequences[i] << std::endl;
        }
        std::cout << "=========================" << std::endl;
    }
};