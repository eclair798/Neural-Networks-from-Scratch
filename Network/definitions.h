#pragma once

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

#include <initializer_list>
#include <string>
#include <vector>
#include <memory>

namespace project {

using DataType = double;

using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Sizes = const std::vector<Index>&;

using FuncAct = std::function<DataType(DataType)>;
using FuncDerivativeAct = std::function<DataType(DataType)>;
using FuncDist = std::function<DataType(const Vector&, const Vector&)>;
using FuncGrad = std::function<RowVector(const Vector&, const Vector&)>;

struct Data {
    Matrix input_vectors;
    Matrix output_vectors;
};

}  // namespace project
