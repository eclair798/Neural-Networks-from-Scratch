#pragma once

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace project {

using DataType = Eigen::MatrixXd::Scalar;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Sizes = std::vector<Index>;


struct Data {
    Matrix input_vectors;
    Matrix output_vectors;
};

struct Parameter {
    Matrix matrix_a;
    Vector vector_b;
};

using Batches = std::vector<Data>;

}  // namespace project
