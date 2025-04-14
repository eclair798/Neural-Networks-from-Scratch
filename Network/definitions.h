#pragma once

#ifdef FORCE_ASSERT
#undef NDEBUG
#endif

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

#include <algorithm>
#include <variant>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace nn {

using DataType = Eigen::MatrixXd::Scalar;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Sizes = std::vector<Index>;

using Path = std::filesystem::path;

struct Data {
    Matrix input_vectors;
    Matrix output_vectors;
};

struct Parameter {
    Matrix matrix_a;
    Vector vector_b;
};

using Batches = std::vector<Data>;

}  // namespace nn
