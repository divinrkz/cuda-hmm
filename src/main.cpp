#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include "hmm.hpp"     // Original CPU implementation header
#include "hmm_gpu.cuh" // GPU implementation header

// -----------------------------------------------------------------------------
// Configuration Parsing (ported from the user-provided code)
// -----------------------------------------------------------------------------
struct HMMConfig
{
    int N, M;                                // states, observations
    std::vector<float> start_p;              // initial probabilities
    std::vector<std::vector<float>> trans_p; // transition matrix
    std::vector<std::vector<float>> emit_p;  // emission matrix
    std::vector<std::vector<int>> sequences; // observation sequences
};

// Parse a configuration file in the same format previously supported
HMMConfig parseConfigFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open config file: " + filename);

    HMMConfig config;
    std::string line;
    std::vector<std::string> lines;

    // Consume non-comment lines
    while (std::getline(file, line))
    {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty() && line[0] != '#')
            lines.push_back(line);
    }

    size_t idx = 0;
    config.N = std::stoi(lines[idx++]);
    config.M = std::stoi(lines[idx++]);

    // start probabilities
    std::istringstream iss_start(lines[idx++]);
    std::string token;
    while (iss_start >> token)
        config.start_p.push_back(std::stof(token));

    // transition matrix
    config.trans_p.resize(config.N);
    for (int i = 0; i < config.N; ++i)
    {
        std::istringstream row(lines[idx++]);
        while (row >> token)
            config.trans_p[i].push_back(std::stof(token));
    }

    // emission matrix
    config.emit_p.resize(config.N);
    for (int i = 0; i < config.N; ++i)
    {
        std::istringstream row(lines[idx++]);
        while (row >> token)
            config.emit_p[i].push_back(std::stof(token));
    }

    // sequences
    std::istringstream seq_info(lines[idx++]);
    int num_sequences, seq_length;
    seq_info >> num_sequences >> seq_length;

    std::vector<int> all_obs;
    for (size_t i = idx; i < lines.size(); ++i)
    {
        std::istringstream row(lines[i]);
        while (row >> token)
            all_obs.push_back(std::stoi(token));
    }

    config.sequences.resize(num_sequences);
    for (int i = 0; i < num_sequences; ++i)
    {
        auto first = all_obs.begin() + i * seq_length;
        auto last = first + seq_length;
        config.sequences[i].assign(first, last);
    }

    return config;
}

// -----------------------------------------------------------------------------
// Helper to flatten 2-D std::vector<float> into 1-D for C-style routines
// -----------------------------------------------------------------------------
static void flatten_vector(const std::vector<std::vector<float>> &mat,
                           std::vector<float> &flat)
{
    flat.clear();
    for (const auto &row : mat)
        flat.insert(flat.end(), row.begin(), row.end());
}

// -----------------------------------------------------------------------------
// Unified driver capable of running either CPU or GPU implementation
// -----------------------------------------------------------------------------
static void run(const HMMConfig &cfg, int problem, int iterations,
                const std::string &impl)
{
    // Pre-flatten matrices for both back-ends
    std::vector<float> A_flat, B_flat;
    flatten_vector(cfg.trans_p, A_flat);
    flatten_vector(cfg.emit_p, B_flat);

    // ---------------- CPU branch ----------------
    if (impl == "cpu")
    {
        IHMM hmm_cpu(cfg.N, cfg.M);

        switch (problem)
        {
        case 1: // Forward
            for (const auto &seq : cfg.sequences)
            {
                std::vector<float> obs_f(seq.begin(), seq.end());
                float **alphas = hmm_cpu.forward(obs_f.data(), nullptr, const_cast<float *>(cfg.start_p.data()), A_flat.data(), B_flat.data(),
                                                 static_cast<int>(seq.size()), cfg.N, cfg.M);
                float total_prob = 0.0f;
                for (int i = 0; i < cfg.N; i++)
                {
                    total_prob += alphas[seq.size()][i];
                }
                std::cout << std::scientific << std::setprecision(6) << total_prob << std::endl;

                for (size_t t = 0; t <= seq.size(); t++)
                {
                    delete[] alphas[t];
                }
                delete[] alphas;
            }
            break;
        case 2: // Viterbi
            for (const auto &seq : cfg.sequences)
            {
                // IHMM expects observations as floats – convert the int sequence
                std::vector<float> obs_f(seq.begin(), seq.end());

                std::string path = hmm_cpu.viterbi(obs_f.data(),
                                                   /*states*/ nullptr,
                                                   const_cast<float *>(cfg.start_p.data()),
                                                   A_flat.data(),
                                                   B_flat.data(),
                                                   static_cast<int>(seq.size()),
                                                   cfg.N, cfg.M);
                std::cout << path << std::endl;
            }
            break;
        case 3:
        { // Baum-Welch (training)
            if (cfg.sequences.empty())
            {
                std::cerr << "No sequences provided in configuration file." << std::endl;
                return;
            }

            // Copies because Baum-Welch updates them in-place
            std::vector<float> A_train = A_flat;
            std::vector<float> B_train = B_flat;
            std::vector<float> pi_train = cfg.start_p;

            const std::vector<int> &obs_int = cfg.sequences[0];
            std::vector<float> obs_f(obs_int.begin(), obs_int.end());

            hmm_cpu.baum_welch(obs_f.data(),
                               /*states*/ nullptr,
                               pi_train.data(),
                               A_train.data(),
                               B_train.data(),
                               static_cast<int>(obs_f.size()),
                               cfg.N,
                               cfg.M,
                               iterations);

            // Output trained parameters (same format expected by test suite)
            std::cout << std::fixed << std::setprecision(6);

            for (int i = 0; i < cfg.N; ++i)
            {
                std::cout << pi_train[i] << (i == cfg.N - 1 ? "" : " ");
            }
            std::cout << std::endl;

            for (int i = 0; i < cfg.N; ++i)
            {
                for (int j = 0; j < cfg.N; ++j)
                {
                    std::cout << A_train[i * cfg.N + j] << (j == cfg.N - 1 ? "" : " ");
                }
                std::cout << std::endl;
            }

            for (int i = 0; i < cfg.N; ++i)
            {
                for (int j = 0; j < cfg.M; ++j)
                {
                    std::cout << B_train[i * cfg.M + j] << (j == cfg.M - 1 ? "" : " ");
                }
                std::cout << std::endl;
            }
            break;
        }
        case 4: // Backward
            for (const auto &seq : cfg.sequences)
            {
                std::vector<float> obs_f(seq.begin(), seq.end());
                float **betas = hmm_cpu.backward(obs_f.data(), nullptr, const_cast<float *>(cfg.start_p.data()), A_flat.data(), B_flat.data(),
                                                 static_cast<int>(seq.size()), cfg.N, cfg.M);

                float total_prob = 0.0f;
                for (int i = 0; i < cfg.N; i++)
                {
                    total_prob += cfg.start_p[i] * B_flat[i * cfg.M + static_cast<int>(obs_f[0])] * betas[1][i];
                }
                std::cout << std::scientific << std::setprecision(6) << total_prob << std::endl;

                for (size_t t = 0; t <= seq.size(); t++)
                {
                    delete[] betas[t];
                }
                delete[] betas;
            }
            break;
        default:
            std::cerr << "Selected problem not supported on CPU path in this driver." << std::endl;
            break;
        }
        return;
    }

    // ---------------- GPU branch ----------------
    if (impl == "gpu")
    {
        int max_T = 0;
        for (const auto &seq : cfg.sequences)
            max_T = std::max(max_T, static_cast<int>(seq.size()));

        HMM_GPU hmm_gpu(cfg.N, cfg.M, max_T);

        if (problem == 1)
        {
            for (const auto &seq : cfg.sequences)
            {
                float prob = hmm_gpu.forward(seq.data(), A_flat.data(), B_flat.data(), cfg.start_p.data(), static_cast<int>(seq.size()));
                std::cout << std::scientific << std::setprecision(6) << prob << std::endl;
            }
        }
        else if (problem == 2)
        { // Viterbi
            for (const auto &seq : cfg.sequences)
            {
                std::string path = hmm_gpu.viterbi(seq.data(),
                                                   A_flat.data(),
                                                   B_flat.data(),
                                                   cfg.start_p.data(),
                                                   static_cast<int>(seq.size()));
                std::cout << path << std::endl;
            }
        }
        else if (problem == 3)
        { // Baum-Welch
            // training vectors (copies, as BW mutates them)
            std::vector<float> A_train = A_flat;
            std::vector<float> B_train = B_flat;
            std::vector<float> pi_train = cfg.start_p;

            if (cfg.sequences.empty())
            {
                std::cerr << "No sequences provided in configuration file." << std::endl;
                return;
            }

            const std::vector<int> &obs = cfg.sequences[0]; // as per original behaviour
            hmm_gpu.baum_welch(obs.data(),
                               A_train.data(),
                               B_train.data(),
                               pi_train.data(),
                               static_cast<int>(obs.size()),
                               iterations,
                               1e-5f);

            // Print trained parameters in a format identical to the previous program
            std::cout << std::fixed << std::setprecision(6);
            for (int i = 0; i < cfg.N; ++i)
            {
                std::cout << pi_train[i] << (i == cfg.N - 1 ? "" : " ");
            }
            std::cout << std::endl;

            for (int i = 0; i < cfg.N; ++i)
            {
                for (int j = 0; j < cfg.N; ++j)
                {
                    std::cout << A_train[i * cfg.N + j]
                              << (j == cfg.N - 1 ? "" : " ");
                }
                std::cout << std::endl;
            }

            for (int i = 0; i < cfg.N; ++i)
            {
                for (int j = 0; j < cfg.M; ++j)
                {
                    std::cout << B_train[i * cfg.M + j]
                              << (j == cfg.M - 1 ? "" : " ");
                }
                std::cout << std::endl;
            }
        }
        else if (problem == 4)
        {
            for (const auto &seq : cfg.sequences)
            {
                float prob = hmm_gpu.backward(seq.data(), A_flat.data(), B_flat.data(), cfg.start_p.data(), static_cast<int>(seq.size()));
                std::cout << std::scientific << std::setprecision(6) << prob << std::endl;
            }
        }
        else
        {
            std::cerr << "Selected problem not implemented on GPU path in this driver." << std::endl;
        }
        return;
    }

    // invalid impl
    std::cerr << "Unknown implementation string (expected 'cpu' or 'gpu')." << std::endl;
}

// -----------------------------------------------------------------------------
// CLI helpers
// -----------------------------------------------------------------------------
static void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog
              << " --impl <cpu|gpu> -c <config_file> -p<problem> [-n <iterations>]" << std::endl;
    std::cout << "  -p1: Forward" << std::endl;
    std::cout << "  -p2: Viterbi" << std::endl;
    std::cout << "  -p3: Baum-Welch" << std::endl;
    std::cout << "  -p4: Backward" << std::endl;
}

int main(int argc, char *argv[])
{
    std::string impl = "cpu";
    std::string cfg_file;
    int problem = 0;
    int iterations = 100;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);
        if (arg == "--impl" && i + 1 < argc)
        {
            impl = argv[++i];
        }
        else if (arg == "-c" && i + 1 < argc)
        {
            cfg_file = argv[++i];
        }
        else if (arg.rfind("-p", 0) == 0)
        {
            problem = std::stoi(arg.substr(2));
        }
        else if (arg == "-n" && i + 1 < argc)
        {
            iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (cfg_file.empty() || problem == 0 || (impl != "cpu" && impl != "gpu"))
    {
        printUsage(argv[0]);
        return 1;
    }

    try
    {
        HMMConfig cfg = parseConfigFile(cfg_file);
        run(cfg, problem, iterations, impl);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}