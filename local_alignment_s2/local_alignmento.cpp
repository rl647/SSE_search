#include <omp.h>
#include <fstream>   // For ifstream and ofstream
#include <sstream> 
#include <iostream>
#include <vector>
#include <unordered_map>
#include <deque>
#include <string>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Required to support std::vector, std::unordered_map, etc.
#include <stdexcept>
namespace py = pybind11;

using SS2Map = std::unordered_map<std::string, std::string>;
using SSR2Map = std::unordered_map<std::string, std::vector<std::pair<int, int>>>;


// Function to read the target database from file
std::unordered_map<std::string, std::string> load_ss2_database(const std::string &filename) {
    std::unordered_map<std::string, std::string> target_ss2;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        size_t pos = line.find('\t');
        if (pos != std::string::npos) {
            target_ss2[line.substr(0, pos)] = line.substr(pos + 1);
        }
    }
    return target_ss2;
}

std::unordered_map<std::string, std::vector<std::pair<int, int>>> load_ssr2_database(const std::string &filename) {
    std::unordered_map<std::string, std::vector<std::pair<int, int>>> target_ssr2;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string key;
        iss >> key;  // Extract the first column as key

        std::vector<std::pair<int, int>> regions;
        std::string region;
        while (iss >> region) {  // Read the remaining tab-separated `start-end` pairs
            size_t dash_pos = region.find('-');
            if (dash_pos != std::string::npos) {
                int start = std::stoi(region.substr(0, dash_pos));
                int end = std::stoi(region.substr(dash_pos + 1));
                regions.emplace_back(start, end);
            }
        }
        target_ssr2[key] = regions;
    }

    return target_ssr2;
}



struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

struct AlignmentResult {
    std::string aligned_ss1;
    std::string alignment_markers;
    std::string aligned_ss2;
    std::string as1;
    std::string as2;
};

AlignmentResult local_alignment(const std::string &ss1, const std::string &ss2,
                                const std::vector<std::pair<int, int>>& ssr1,
                                const std::vector<std::pair<int, int>>& ssr2,
                                const std::unordered_map<char, std::unordered_map<char, double>> &similarity_matrix) {
    // Define directions for traceback
    const std::pair<int, int> DIAG = {-1, -1};
    const std::pair<int, int> LEFT = {-1, 0};
    const std::pair<int, int> UP = {0, -1};
    const std::vector<std::pair<int, int>> option_pos = {DIAG, DIAG, LEFT, UP};
    std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash> pos;

    size_t len1 = ss1.size(), len2 = ss2.size();
    std::vector<std::vector<double>> H(len1 + 1, std::vector<double>(len2 + 1, 0));
    std::vector<std::vector<double>> X(len1 + 1, std::vector<double>(len2 + 1, 0));
    std::vector<std::vector<double>> Y(len1 + 1, std::vector<double>(len2 + 1, 0));
    std::vector<std::vector<double>> M(len1 + 1, std::vector<double>(len2 + 1, 0));

    // Fill score matrices using dynamic programming
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {

            H[i][j] = similarity_matrix.at(ss1[i - 1]).at(ss2[j - 1]) +
                      std::max({H[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]});


            X[i][j] = std::max({
                similarity_matrix.at(ss1[i - 1]).at('-') + H[i][j - 1],
                similarity_matrix.at(ss1[i - 1]).at('=') + X[i][j - 1],
                0.0
            });
            
            Y[i][j] = std::max({
                similarity_matrix.at(ss2[j - 1]).at('-') + H[i - 1][j],
                similarity_matrix.at(ss2[j - 1]).at('=') + Y[i - 1][j],
                0.0
            });

            std::vector<double> options = {0, H[i][j], Y[i][j], X[i][j]};
            double max_score = *std::max_element(options.begin(), options.end());
            M[i][j] = max_score;

            size_t max_index = std::distance(options.begin(), std::max_element(options.begin(), options.end()));
            pos[{i, j}] = option_pos[max_index];
        }
    }

    // Find the optimal score for starting the traceback
    size_t i = 0, j = 0;
    double max_value = 0;
    for (size_t x = 0; x <= len1; ++x) {
        for (size_t y = 0; y <= len2; ++y) {
            if (M[x][y] > max_value) {
                max_value = M[x][y];
                i = x;
                j = y;
            }
        }
    }

    // Traceback to find the optimal alignment
    std::deque<std::pair<int, int>> alignment;
    while (i >= 0 && j >= 0 && M[i][j] > 0) {
        auto direction = pos[{i, j}];
        if (direction == DIAG) {
            alignment.emplace_front(i - 1, j - 1);
        } else if (direction == LEFT) {
            alignment.emplace_front(i - 1, -1);
        } else if (direction == UP) {
            alignment.emplace_front(-1, j - 1);
        }
        int di = direction.first;
        int dj = direction.second;
        i += di;
        j += dj;
    }

    // Prepare alignment strings
    AlignmentResult result;
    for (const auto &[i, e] : alignment) {
        if (i == -1) {
            result.aligned_ss1 += "_0_-";
            result.alignment_markers += ' ';
            if (e>=1){
                 result.aligned_ss2 += std::string("_")+std::to_string(ssr2[e].first - ssr2[e - 1].second);
            } else{
                 result.aligned_ss2 += "_0";
            }
           
            result.aligned_ss2 += std::string("_") + ss2[e];
            result.as2 +=  ss2[e];
        } else if (e == -1) {
            if (i>=1){
               result.aligned_ss1 += std::string("_")+std::to_string(ssr1[i].first - ssr1[i - 1].second);
            } else{
                 result.aligned_ss1 += "_0";
            }
            result.aligned_ss1 += std::string("_") + ss1[i];
            result.as1 +=  ss1[i];

            result.alignment_markers += ' ';
            result.aligned_ss2 += "_0_-";
        } else {
            if (i>=1){
               result.aligned_ss1 += std::string("_")+std::to_string(ssr1[i].first - ssr1[i - 1].second);
            } else{
                 result.aligned_ss1 += "_0";
            }
            result.aligned_ss1 += std::string("_") + ss1[i];
            result.as1 +=  ss1[i];
            result.as2 +=  ss2[e];
            if (e>=1){
                 result.aligned_ss2 += std::string("_")+std::to_string(ssr2[e].first - ssr2[e - 1].second);
            } else{
                 result.aligned_ss2 += "_0";
            } 
            result.aligned_ss2 += std::string("_") +ss2[e];
            if (ss1[i] == ss2[e]) {
                result.alignment_markers += '|';
            } else if (similarity_matrix.at(ss1[i]).at(ss2[e]) > 0) {
                result.alignment_markers += ':';
            } else {
                result.alignment_markers += '.';
            }
        }
    }

    return result;
    
}
void align(const std::string &ss1_db_path,
           const std::string &ssr1_db_path,
           const std::string &ss2_db_path,
           const std::string &ssr2_db_path,
           const std::string &results_path,
           const std::unordered_map<char, std::unordered_map<char, double>> &similarity_matrix) {

    // Load target databases
    auto query_ss1 = load_ss2_database(ss1_db_path);
    auto query_ssr1 = load_ssr2_database(ssr1_db_path);
    auto target_ss2 = load_ss2_database(ss2_db_path);
    auto target_ssr2 = load_ssr2_database(ssr2_db_path);

    // Convert query_ss1 to a vector of keys for parallelization
    std::vector<std::string> query_keys;
    for (const auto &[key, _] : query_ss1) {
        query_keys.push_back(key);
    }

    // Open output file
    #pragma omp parallel for
    for (size_t i = 0; i < query_keys.size(); ++i) {
        const auto &key1 = query_keys[i];
        const auto &ss1 = query_ss1.at(key1);
        
        std::string rp = results_path + "/" + key1 + ".txt"; 
        std::vector<std::pair<int, int>> ssr1 = query_ssr1.at(key1);
        std::ofstream outfile(rp, std::ios::app);
        if (!outfile) {
            throw std::runtime_error("Could not open results file.");
        }
        
        // Iterate through database and align
        for (const auto &[key, ss2] : target_ss2) {
            if (target_ssr2.find(key) == target_ssr2.end()) continue;
            std::vector<std::pair<int, int>> ssr2 = target_ssr2.at(key);
            if (ss2.empty() || ssr2.empty()) {
                std::cout << "ss1: " << ss1 << std::endl;
                std::cout << "ss2: " << ss2 << std::endl;
                continue;
            }
            if (key1 != key) {
                AlignmentResult result = local_alignment(ss1, ss2, ssr1, ssr2, similarity_matrix);
                int ms = std::min(result.as1.length(), result.as2.length()); 
                int ms2 = std::max(result.as1.length(), result.as2.length()); 
            
                if ((((ms2 - ms) / static_cast<double>(ms) <= 0.6 && ms <= 10) || 
                    ((ms2 - ms) / static_cast<double>(ms) <= 0.5 && ms > 10)) && ms>=5) {
                    outfile << key << "\t" << result.aligned_ss1 << "\t" << result.aligned_ss2 << "\n";
                }
            }
        }
        outfile.close();
    }
}

// Expose function to Python
PYBIND11_MODULE(local_alignment, m) {
    m.def("align", &align, "Perform local alignment",
          py::arg("ss1_db_path"), py::arg("ssr1_db_path"),
          py::arg("ss2_db_path"), py::arg("ssr2_db_path"),
          py::arg("results_path"), py::arg("similarity_matrix"));
}
