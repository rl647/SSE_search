#include <iostream>
#include <vector>
#include <unordered_map>
#include <deque>
#include <string>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Custom hash function for std::pair<int, int>
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
    double score;
    double normalized_score;
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
    double self_score_ss1 = 0.0;
    // for (char c : ss1) {
    //     self_score_ss1 += similarity_matrix.at(c).at(c);  // 计算 ss1 的自对齐分数
    // }
    
    // // 添加对 ss2 的循环计算
    double self_score_ss2 = 0.0;
    // for (char c : ss2) {  // 遍历 ss2 的每个字符
    //     self_score_ss2 += similarity_matrix.at(c).at(c);  // 累加分数
    // }
    
    
    // // 计算归一化分数
    // double max_self_score = std::max(self_score_ss1, self_score_ss2);
    // double normalized_score = (max_self_score != 0) ? (max_value / max_self_score) : 0.0;

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
            self_score_ss2 += similarity_matrix.at(ss2[e]).at(ss2[e]);
        } else if (e == -1) {
            if (i>=1){
               result.aligned_ss1 += std::string("_")+std::to_string(ssr1[i].first - ssr1[i - 1].second);
            } else{
                 result.aligned_ss1 += "_0";
            }
            result.aligned_ss1 += std::string("_") + ss1[i];
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
    double max_self_score = std::max(self_score_ss1, self_score_ss2);
    double normalized_score = (max_self_score != 0) ? (max_value / max_self_score) : 0.0;
    result.score = max_value;
    result.normalized_score = normalized_score;
    return result;
}

PYBIND11_MODULE(local_alignment, m) {
    m.doc() = "Local Alignment Module";

    // Expose the AlignmentResult struct to Python
    py::class_<AlignmentResult>(m, "AlignmentResult")
        .def_readwrite("aligned_ss1", &AlignmentResult::aligned_ss1)
        .def_readwrite("alignment_markers", &AlignmentResult::alignment_markers)
        .def_readwrite("aligned_ss2", &AlignmentResult::aligned_ss2)
        .def_readwrite("score", &AlignmentResult::score)
        .def_readwrite("normalized_score", &AlignmentResult::normalized_score);

    // Corrected function binding to match the function signature exactly
    m.def("local_alignment", &local_alignment, "Local alignment of two sequences",
          py::arg("ss1"), py::arg("ss2"), py::arg("ssr1"), py::arg("ssr2"), py::arg("similarity_matrix"));
}













