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
using Sizes = const std::vector<Index>&;

using Counter = int;

using FuncAct = std::function<Vector(Vector)>;
using FuncDerivativeAct = std::function<Matrix(Vector)>;
using FuncDerivativeDim1Act = std::function<DataType(DataType)>;

using FuncDist = std::function<DataType(const Vector&, const Vector&)>;
using FuncGrad = std::function<RowVector(const Vector&, const Vector&)>;

using Path = const std::string&;

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
