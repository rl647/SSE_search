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
#include <iomanip> 
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

struct AlignmentResult_v2 {
    std::string aligned_ss1;
    std::string alignment_markers;
    std::string aligned_ss2;
    std::string as1;
    std::string as2;
    double score;
    double normalized_score;
};

AlignmentResult_v2 local_alignment(const std::string &ss1, const std::string &ss2,
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


    double self_score_ss1 = 0.0;
    // std::cout << "Score: " << self_score_ss1 << std::endl;
    // for (char c : ss1) {
    //     self_score_ss1 += similarity_matrix.at(c).at(c);  // 计算 ss1 的自对齐分数
    // }
    
    // // 添加对 ss2 的循环计算
    double self_score_ss2 = 0.0;
    // for (char c : ss2) {  // 遍历 ss2 的每个字符
    //     self_score_ss2 += similarity_matrix.at(c).at(c);  // 累加分数
    // }
    // double max_self_score = std::max(self_score_ss1, self_score_ss2);
    // double normalized_score = (max_self_score != 0) ? (max_value / max_self_score) : 0.0;
    // Prepare alignment strings
    AlignmentResult_v2 result;
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
            self_score_ss2 += similarity_matrix.at(ss2[e]).at(ss2[e]);
            result.as2 +=  ss2[e];
        } else if (e == -1) {
            if (i>=1){
               result.aligned_ss1 += std::string("_")+std::to_string(ssr1[i].first - ssr1[i - 1].second);
            } else{
                 result.aligned_ss1 += "_0";
            }
            result.aligned_ss1 += std::string("_") + ss1[i];
            result.as1 +=  ss1[i];
            self_score_ss1 += similarity_matrix.at(ss1[i]).at(ss1[i]);

            result.alignment_markers += ' ';
            result.aligned_ss2 += "_0_-";
        } else {
            if (i>=1){
               result.aligned_ss1 += std::string("_")+std::to_string(ssr1[i].first - ssr1[i - 1].second);
            } else{
                 result.aligned_ss1 += "_0";
            }
            result.aligned_ss1 += std::string("_") + ss1[i];
            self_score_ss1 += similarity_matrix.at(ss1[i]).at(ss1[i]);

            result.as1 +=  ss1[i];
            result.as2 +=  ss2[e];
            if (e>=1){
                 result.aligned_ss2 += std::string("_")+std::to_string(ssr2[e].first - ssr2[e - 1].second);
            } else{
                 result.aligned_ss2 += "_0";
            } 
            result.aligned_ss2 += std::string("_") +ss2[e];
            self_score_ss2 += similarity_matrix.at(ss2[e]).at(ss2[e]);

            if (ss1[i] == ss2[e]) {
                result.alignment_markers += '|';
            } else if (similarity_matrix.at(ss1[i]).at(ss2[e]) > 0) {
                result.alignment_markers += ':';
            } else {
                result.alignment_markers += '.';
            }
        }
    }
    // 计算归一化分数
    double max_self_score = std::max(self_score_ss1, self_score_ss2);
    double normalized_score = (max_self_score != 0) ? (max_value / max_self_score) : 0.0;
    result.score = max_value;
    result.normalized_score = normalized_score;
    // std::cout << "Score: " << max_value << std::endl;
    return result;
    
}
void align(const std::string &ss1_db_path,
           const std::string &ssr1_db_path,
           const std::string &ss2_db_path,
           const std::string &ssr2_db_path,
           const std::string &results_path,
           const std::unordered_map<char, std::unordered_map<char, double>> &similarity_matrix,
           const float threshold) {

    // Load target databases
    auto query_ss1 = load_ss2_database(ss1_db_path);
    auto query_ssr1 = load_ssr2_database(ssr1_db_path);
    auto target_ss2 = load_ss2_database(ss2_db_path);
    auto target_ssr2 = load_ssr2_database(ssr2_db_path);
    // std::cout << "Loaded " << query_ss1.size() << " query sequences." << std::endl;
    // std::cout << ">>> target_ss2.size() = " << target_ss2.size()
    // << ", target_ssr2.size() = " << target_ssr2.size()
    // << std::endl;
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
        // std::cout << "Processing " << key1 << std::endl;
        std::string rp = results_path + "/" + key1 + ".txt"; 
        std::vector<std::pair<int, int>> ssr1 = query_ssr1.at(key1);
        std::ofstream outfile(rp, std::ios::app);
        if (!outfile) {
            throw std::runtime_error("Could not open results file.");
        }
        // std::cout << "Processing ss1 " << key1 << std::endl;
        // Iterate through database and align
        for (const auto &[key, ss2] : target_ss2) {
            // std::cout << "Processing ss2 " << key << std::endl;
            if (target_ssr2.find(key) == target_ssr2.end()) continue;
            std::vector<std::pair<int, int>> ssr2 = target_ssr2.at(key);
            if (ss2.empty() || ssr2.empty()) {
                std::cout << "ss1: " << ss1 << std::endl;
                std::cout << "ss2: " << ss2 << std::endl;
                continue;
            }
            
            if (key1 != key) {
                AlignmentResult_v2 result = local_alignment(ss1, ss2, ssr1, ssr2, similarity_matrix);
                int ms = std::min(result.as1.length(), result.as2.length()); 
               
                if (ms>=5){
                    double l1 = static_cast<double>(result.as1.length()) / ss1.length();
                    double l2 = static_cast<double>(result.as2.length()) / ss2.length();
                    double l = std::max(l1, l2);
                    double exponent = 1.0 / std::sqrt(std::max(l, 1e-4));
                    double score = std::max(result.normalized_score, 1e-6);
                    result.normalized_score = std::pow(score,exponent);
                    
                    if (result.normalized_score >= threshold) {
                        result.normalized_score = std::pow(score, exponent);
                        
                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(3) << result.normalized_score;
                        std::string ns = oss.str();
                        
                        outfile << key << "\t" << result.aligned_ss1 << "\t" << result.aligned_ss2 << "\t" << ns << "\n";
                    }
            }
                
            }
        }
        outfile.close();
    }
}

PYBIND11_MODULE(local_alignment_v2, m) {
    py::class_<AlignmentResult_v2>(m, "AlignmentResult_v2")
        .def_readwrite("aligned_ss1", &AlignmentResult_v2::aligned_ss1)
        .def_readwrite("alignment_markers", &AlignmentResult_v2::alignment_markers)
        .def_readwrite("aligned_ss2", &AlignmentResult_v2::aligned_ss2)
        .def_readwrite("as1", &AlignmentResult_v2::as1)
        .def_readwrite("as2", &AlignmentResult_v2::as2)
        .def_readwrite("score", &AlignmentResult_v2::score)
        .def_readwrite("normalized_score", &AlignmentResult_v2::normalized_score);

    m.def("align", &align, "Perform local alignment",
          py::arg("ss1_db_path"), py::arg("ssr1_db_path"),
          py::arg("ss2_db_path"), py::arg("ssr2_db_path"),
          py::arg("results_path"), py::arg("similarity_matrix"),py::arg("threshold") = 0.5 );
}
