#pragma once

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace project {

using DataType = double;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Sizes = const std::vector<Index>&;

using Counter = int;

using FuncAct = std::function<DataType(DataType)>;
using FuncDerivativeAct = std::function<DataType(DataType)>;
using FuncDist = std::function<DataType(const Vector&, const Vector&)>;
using FuncGrad = std::function<RowVector(const Vector&, const Vector&)>;

struct Data {
    Matrix input_vectors;
    Matrix output_vectors;
};
using Batches = std::vector<Data>;

}  // namespace project
