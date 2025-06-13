#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include "hmm.hpp"
#include "hmm_gpu.cuh"

// Re-use the same HMMConfig struct and helper from main.cpp
struct HMMConfig {
    int N; int M;
    std::vector<float> start_p;
    std::vector<std::vector<float>> trans_p;
    std::vector<std::vector<float>> emit_p;
    std::vector<std::vector<int>> sequences;
};

// ---------------- Utility parsing & CPU helpers (copied from main.cpp) -----------------

HMMConfig parseConfigFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open config file: " + filename);
    }

    HMMConfig config;
    std::string line;
    std::vector<std::string> lines;

    while (std::getline(file, line))
    {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty() && line[0] != '#')
        {
            lines.push_back(line);
        }
    }

    size_t idx = 0;
    config.N = std::stoi(lines[idx++]);
    config.M = std::stoi(lines[idx++]);

    std::istringstream iss(lines[idx++]);
    std::string token;
    while (iss >> token) config.start_p.push_back(std::stof(token));

    config.trans_p.resize(config.N);
    for (int i = 0; i < config.N; i++)
    {
        std::istringstream row(lines[idx++]);
        while (row >> token) config.trans_p[i].push_back(std::stof(token));
    }

    config.emit_p.resize(config.N);
    for (int i = 0; i < config.N; i++)
    {
        std::istringstream row(lines[idx++]);
        while (row >> token) config.emit_p[i].push_back(std::stof(token));
    }

    std::istringstream seq_info(lines[idx++]);
    int num_sequences, seq_length;
    seq_info >> num_sequences >> seq_length;

    std::vector<int> all_obs;
    for (size_t i = idx; i < lines.size(); i++)
    {
        std::istringstream row(lines[i]);
        while (row >> token) all_obs.push_back(std::stoi(token));
    }

    config.sequences.resize(num_sequences);
    for (int i = 0; i < num_sequences; i++)
    {
        config.sequences[i].resize(seq_length);
        for (int j = 0; j < seq_length; j++)
            config.sequences[i][j] = all_obs[i * seq_length + j];
    }

    return config;
}

void runForwardAlgorithm(const HMMConfig &config)
{
    IHMM hmm(config.N, config.M);

    std::vector<float> start_p_v(config.start_p.begin(), config.start_p.end());
    std::vector<float> trans_p_v(config.N * config.N);
    std::vector<float> emit_p_v(config.N * config.M);

    for (int i = 0; i < config.N; i++)
        for (int j = 0; j < config.N; j++)
            trans_p_v[i * config.N + j] = config.trans_p[i][j];

    for (int i = 0; i < config.N; i++)
        for (int j = 0; j < config.M; j++)
            emit_p_v[i * config.M + j] = config.emit_p[i][j];

    for (const auto &sequence : config.sequences)
    {
        std::vector<float> obs(sequence.begin(), sequence.end());
        float **alphas = hmm.forward(obs.data(), nullptr, start_p_v.data(), trans_p_v.data(), emit_p_v.data(),
                                     (int)sequence.size(), config.N, config.M);
        float total_prob = 0.0f;
        for (int i = 0; i < config.N; i++) total_prob += alphas[sequence.size()][i];
        std::cout << std::scientific << std::setprecision(6) << total_prob << std::endl;
        for (size_t t = 0; t <= sequence.size(); t++) delete[] alphas[t];
        delete[] alphas;
    }
}

void runViterbiAlgorithm(const HMMConfig &config)
{
    IHMM hmm(config.N, config.M);
    std::vector<float> start_p_v(config.start_p.begin(), config.start_p.end());
    std::vector<float> trans_p_v(config.N * config.N);
    std::vector<float> emit_p_v(config.N * config.M);
    for (int i = 0; i < config.N; i++)
        for (int j = 0; j < config.N; j++)
            trans_p_v[i * config.N + j] = config.trans_p[i][j];
    for (int i = 0; i < config.N; i++)
        for (int j = 0; j < config.M; j++)
            emit_p_v[i * config.M + j] = config.emit_p[i][j];

    for (const auto &sequence : config.sequences)
    {
        std::vector<float> obs(sequence.begin(), sequence.end());
        std::string path = hmm.viterbi(obs.data(), nullptr, start_p_v.data(), trans_p_v.data(), emit_p_v.data(),
                                       (int)sequence.size(), config.N, config.M);
        std::cout << path << std::endl;
    }
}

void runBackwardAlgorithm(const HMMConfig &config)
{
    IHMM hmm(config.N, config.M);
    std::vector<float> start_p_v(config.start_p.begin(), config.start_p.end());
    std::vector<float> trans_p_v(config.N * config.N);
    std::vector<float> emit_p_v(config.N * config.M);
    for (int i = 0; i < config.N; i++)
        for (int j = 0; j < config.N; j++)
            trans_p_v[i * config.N + j] = config.trans_p[i][j];
    for (int i = 0; i < config.N; i++)
        for (int j = 0; j < config.M; j++)
            emit_p_v[i * config.M + j] = config.emit_p[i][j];

    for (const auto &sequence : config.sequences)
    {
        std::vector<float> obs(sequence.begin(), sequence.end());
        float **betas = hmm.backward(obs.data(), nullptr, start_p_v.data(), trans_p_v.data(), emit_p_v.data(),
                                     (int)sequence.size(), config.N, config.M);
        float total_prob = 0.0f;
        for (int i = 0; i < config.N; i++)
            total_prob += start_p_v[i] * emit_p_v[i * config.M + (int)obs[0]] * betas[1][i];
        std::cout << std::scientific << std::setprecision(6) << total_prob << std::endl;
        for (size_t t = 0; t <= sequence.size(); t++) delete[] betas[t];
        delete[] betas;
    }
}

// --- GPU Baum-Welch variant ---
void runBaumWelchAlgorithmGPU(const HMMConfig &cfg, int iterations) {
    // Host arrays
    std::vector<float> h_A(cfg.N * cfg.N);
    std::vector<float> h_B(cfg.N * cfg.M);
    std::vector<float> h_pi(cfg.N);

    for(int i=0;i<cfg.N;i++) h_pi[i] = cfg.start_p[i];
    for(int i=0;i<cfg.N;i++)
        for(int j=0;j<cfg.N;j++)
            h_A[i*cfg.N + j] = cfg.trans_p[i][j];
    for(int i=0;i<cfg.N;i++)
        for(int k=0;k<cfg.M;k++)
            h_B[i*cfg.M + k] = cfg.emit_p[i][k];

    if(cfg.sequences.empty()) {
        std::cerr << "No sequences in config file" << std::endl;
        return;
    }
    const std::vector<int>& seq = cfg.sequences[0];
    std::vector<int> h_obs(seq.begin(), seq.end());

    // Train on GPU
    baumWelchGPU(h_A.data(), h_B.data(), h_pi.data(), h_obs.data(),
                 cfg.N, cfg.M, (int)seq.size(), iterations, 1e-4f);

    // Print parameters in same format as CPU version
    std::cout << std::fixed << std::setprecision(6);
    // Initial probabilities
    for(int i=0;i<cfg.N;i++) {
        std::cout << h_pi[i];
        if(i < cfg.N-1) std::cout << " ";
    }
    std::cout << "\n";

    // Transition matrix
    for(int i=0;i<cfg.N;i++) {
        for(int j=0;j<cfg.N;j++) {
            std::cout << h_A[i*cfg.N + j];
            if(j < cfg.N-1) std::cout << " ";
        }
        std::cout << "\n";
    }

    // Emission matrix
    for(int i=0;i<cfg.N;i++) {
        for(int k=0;k<cfg.M;k++) {
            std::cout << h_B[i*cfg.M + k];
            if(k < cfg.M-1) std::cout << " ";
        }
        std::cout << "\n";
    }
}

void printUsage(const char* prog){
    std::cout << "Usage: " << prog << " -c <config_file> -p<problem> [-n <iters>]" << std::endl;
}

int main(int argc,char* argv[]){
    std::string config_file; int problem=0; int iterations=100;
    for(int i=1;i<argc;i++){
        if(std::strcmp(argv[i],"-c")==0 && i+1<argc) config_file = argv[++i];
        else if(std::strncmp(argv[i],"-p",2)==0) problem = std::atoi(argv[i]+2);
        else if(std::strcmp(argv[i],"-n")==0 && i+1<argc) iterations = std::atoi(argv[++i]);
    }
    if(config_file.empty()||problem==0){ printUsage(argv[0]); return 1; }

    try{
        HMMConfig cfg = parseConfigFile(config_file);
        switch(problem){
            case 1: runForwardAlgorithm(cfg); break;
            case 2: runViterbiAlgorithm(cfg); break;
            case 3: runBaumWelchAlgorithmGPU(cfg, iterations); break;
            case 4: runBackwardAlgorithm(cfg); break;
            default: std::cerr << "Invalid problem" << std::endl; return 1;
        }
    }catch(const std::exception& e){ std::cerr << "Error: " << e.what() << std::endl; return 1; }
    return 0;
}

// Bring in implementations from main.cpp (compiled into hmm_cpu_lib)
// so we only declare them here. The definitions are linked from that library.