#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include "hmm.hpp"

struct HMMConfig
{
    int N; // number of states
    int M; // number of observations
    std::vector<float> start_p;
    std::vector<std::vector<float>> trans_p;
    std::vector<std::vector<float>> emit_p;
    std::vector<std::vector<int>> sequences;
};

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

    // Read all non-comment, non-empty lines
    while (std::getline(file, line))
    {
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (!line.empty() && line[0] != '#')
        {
            lines.push_back(line);
        }
    }

    size_t idx = 0;

    // Parse number of states
    config.N = std::stoi(lines[idx++]);

    // Parse number of observations
    config.M = std::stoi(lines[idx++]);

    // Parse initial state probabilities
    std::istringstream iss(lines[idx++]);
    std::string token;
    while (iss >> token)
    {
        config.start_p.push_back(std::stof(token));
    }

    // Parse transition matrix
    config.trans_p.resize(config.N);
    for (int i = 0; i < config.N; i++)
    {
        std::istringstream iss(lines[idx++]);
        while (iss >> token)
        {
            config.trans_p[i].push_back(std::stof(token));
        }
    }

    // Parse emission matrix
    config.emit_p.resize(config.N);
    for (int i = 0; i < config.N; i++)
    {
        std::istringstream iss(lines[idx++]);
        while (iss >> token)
        {
            config.emit_p[i].push_back(std::stof(token));
        }
    }

    // Parse sequence info
    std::istringstream seq_info(lines[idx++]);
    int num_sequences, seq_length;
    seq_info >> num_sequences >> seq_length;

    // Parse observation sequences
    std::vector<int> all_obs;
    for (size_t i = idx; i < lines.size(); i++)
    {
        std::istringstream iss(lines[i]);
        while (iss >> token)
        {
            all_obs.push_back(std::stoi(token));
        }
    }

    // Split into individual sequences
    config.sequences.resize(num_sequences);
    for (int i = 0; i < num_sequences; i++)
    {
        config.sequences[i].resize(seq_length);
        for (int j = 0; j < seq_length; j++)
        {
            config.sequences[i][j] = all_obs[i * seq_length + j];
        }
    }

    return config;
}

void runForwardAlgorithm(const HMMConfig &config)
{
    IHMM hmm(config.N, config.M);

    // Convert data structures to arrays
    float *start_p = new float[config.N];
    float *trans_p = new float[config.N * config.N];
    float *emit_p = new float[config.N * config.M];

    for (int i = 0; i < config.N; i++)
    {
        start_p[i] = config.start_p[i];
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.N; j++)
        {
            trans_p[i * config.N + j] = config.trans_p[i][j];
        }
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.M; j++)
        {
            emit_p[i * config.M + j] = config.emit_p[i][j];
        }
    }

    // Process each sequence
    for (const auto &sequence : config.sequences)
    {
        float *obs = new float[sequence.size()];
        for (size_t i = 0; i < sequence.size(); i++)
        {
            obs[i] = static_cast<float>(sequence[i]);
        }

        float **alphas = hmm.forward(obs, nullptr, start_p, trans_p, emit_p,
                                     sequence.size(), config.N, config.M);

        // Calculate total probability
        float total_prob = 0.0f;
        for (int i = 0; i < config.N; i++)
        {
            total_prob += alphas[sequence.size()][i];
        }

        std::cout << std::scientific << std::setprecision(6) << total_prob << std::endl;

        // Clean up alphas
        for (size_t t = 0; t <= sequence.size(); t++)
        {
            delete[] alphas[t];
        }
        delete[] alphas;
        delete[] obs;
    }

    delete[] start_p;
    delete[] trans_p;
    delete[] emit_p;
}

void runViterbiAlgorithm(const HMMConfig &config)
{
    IHMM hmm(config.N, config.M);

    // Convert data structures to arrays
    float *start_p = new float[config.N];
    float *trans_p = new float[config.N * config.N];
    float *emit_p = new float[config.N * config.M];

    for (int i = 0; i < config.N; i++)
    {
        start_p[i] = config.start_p[i];
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.N; j++)
        {
            trans_p[i * config.N + j] = config.trans_p[i][j];
        }
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.M; j++)
        {
            emit_p[i * config.M + j] = config.emit_p[i][j];
        }
    }

    // Process each sequence
    for (const auto &sequence : config.sequences)
    {
        float *obs = new float[sequence.size()];
        for (size_t i = 0; i < sequence.size(); i++)
        {
            obs[i] = static_cast<float>(sequence[i]);
        }

        std::string path = hmm.viterbi(obs, nullptr, start_p, trans_p, emit_p,
                                       sequence.size(), config.N, config.M);

        std::cout << path << std::endl;

        delete[] obs;
    }

    delete[] start_p;
    delete[] trans_p;
    delete[] emit_p;
}

void runBaumWelchAlgorithm(const HMMConfig &config, int iterations)
{
    IHMM hmm(config.N, config.M);

    // Convert data structures to arrays
    float *start_p = new float[config.N];
    float *trans_p = new float[config.N * config.N];
    float *emit_p = new float[config.N * config.M];

    for (int i = 0; i < config.N; i++)
    {
        start_p[i] = config.start_p[i];
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.N; j++)
        {
            trans_p[i * config.N + j] = config.trans_p[i][j];
        }
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.M; j++)
        {
            emit_p[i * config.M + j] = config.emit_p[i][j];
        }
    }

    // Train on all sequences (using first sequence for now - you may want to modify this)
    if (!config.sequences.empty())
    {
        float *obs = new float[config.sequences[0].size()];
        for (size_t i = 0; i < config.sequences[0].size(); i++)
        {
            obs[i] = static_cast<float>(config.sequences[0][i]);
        }

        hmm.baum_welch(obs, nullptr, start_p, trans_p, emit_p,
                       config.sequences[0].size(), config.N, config.M, iterations);

        delete[] obs;
    }

    // Output trained parameters
    // Initial probabilities
    for (int i = 0; i < config.N; i++)
    {
        std::cout << std::fixed << std::setprecision(6) << start_p[i];
        if (i < config.N - 1)
            std::cout << " ";
    }
    std::cout << std::endl;

    // Transition matrix
    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.N; j++)
        {
            std::cout << std::fixed << std::setprecision(6) << trans_p[i * config.N + j];
            if (j < config.N - 1)
                std::cout << " ";
        }
        std::cout << std::endl;
    }

    // Emission matrix
    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.M; j++)
        {
            std::cout << std::fixed << std::setprecision(6) << emit_p[i * config.M + j];
            if (j < config.M - 1)
                std::cout << " ";
        }
        std::cout << std::endl;
    }

    delete[] start_p;
    delete[] trans_p;
    delete[] emit_p;
}

void runBackwardAlgorithm(const HMMConfig &config)
{
    IHMM hmm(config.N, config.M);

    // Convert data structures to arrays
    float *start_p = new float[config.N];
    float *trans_p = new float[config.N * config.N];
    float *emit_p = new float[config.N * config.M];

    for (int i = 0; i < config.N; i++)
    {
        start_p[i] = config.start_p[i];
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.N; j++)
        {
            trans_p[i * config.N + j] = config.trans_p[i][j];
        }
    }

    for (int i = 0; i < config.N; i++)
    {
        for (int j = 0; j < config.M; j++)
        {
            emit_p[i * config.M + j] = config.emit_p[i][j];
        }
    }

    // Process each sequence
    for (const auto &sequence : config.sequences)
    {
        float *obs = new float[sequence.size()];
        for (size_t i = 0; i < sequence.size(); i++)
        {
            obs[i] = static_cast<float>(sequence[i]);
        }

        float **betas = hmm.backward(obs, nullptr, start_p, trans_p, emit_p,
                                     sequence.size(), config.N, config.M);

        // Calculate total probability using backward algorithm
        // P(observations) = sum over all states i of: P(start_i) * P(obs[0]|state_i) * beta[1][i]
        float total_prob = 0.0f;
        for (int i = 0; i < config.N; i++)
        {
            total_prob += start_p[i] * emit_p[i * config.M + static_cast<int>(obs[0])] * betas[1][i];
        }

        std::cout << std::scientific << std::setprecision(6) << total_prob << std::endl;

        // Clean up betas
        for (size_t t = 0; t <= sequence.size(); t++)
        {
            delete[] betas[t];
        }
        delete[] betas;
        delete[] obs;
    }

    delete[] start_p;
    delete[] trans_p;
    delete[] emit_p;
}

void printUsage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " -c <config_file> -p<problem> [-n <iterations>]" << std::endl;
    std::cout << "  -c <config_file>: Path to HMM configuration file" << std::endl;
    std::cout << "  -p1: Forward algorithm (compute probabilities)" << std::endl;
    std::cout << "  -p2: Viterbi algorithm (find most probable sequences)" << std::endl;
    std::cout << "  -p3: Baum-Welch algorithm (train parameters)" << std::endl;
    std::cout << "  -p4: Backward algorithm (compute probabilities)" << std::endl;
    std::cout << "  -n <iterations>: Number of iterations for Baum-Welch (default: 100)" << std::endl;
}

int main(int argc, char *argv[])
{
    std::string config_file;
    int problem = 0;
    int iterations = 100;

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc)
        {
            config_file = argv[++i];
        }
        else if (std::strncmp(argv[i], "-p", 2) == 0)
        {
            problem = std::atoi(argv[i] + 2);
        }
        else if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            iterations = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (config_file.empty() || problem == 0)
    {
        printUsage(argv[0]);
        return 1;
    }

    try
    {
        HMMConfig config = parseConfigFile(config_file);

        switch (problem)
        {
        case 1:
            runForwardAlgorithm(config);
            break;
        case 2:
            runViterbiAlgorithm(config);
            break;
        case 3:
            runBaumWelchAlgorithm(config, iterations);
            break;
        case 4:
            runBackwardAlgorithm(config);
            break;
        default:
            std::cerr << "Invalid problem number. Use 1, 2, 3, or 4." << std::endl;
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}