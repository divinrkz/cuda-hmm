#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include "hmm.hpp"

class HMMDataLoader {
public:
    struct HMMData {
        std::vector<std::vector<float>> transition_matrix;
        std::vector<std::vector<float>> emission_matrix;
        std::vector<std::vector<int>> sequences;
        int N;
        int M;
    };

    static HMMData load_sequence(int n);
    static void sequence_prediction(int n);
}; 