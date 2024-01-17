#pragma once

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

#include <initializer_list>
#include <string>
#include <vector>
#include <memory>

namespace project {

using DataType = double;


using Sizes = std::initializer_list<size_t>;

using Matrix = Eigen::MatrixXd;
using Index = Matrix::Index;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

using Data = std::pair<Matrix, Matrix>;

using FuncName = std::string;
using FuncNames = std::initializer_list<FuncName>;

using FuncAct = std::function<double(double)>;
using FuncDist = std::function<double(RowVector, RowVector)>;
using FuncDerivativeDist = std::function<RowVector(RowVector, RowVector)>;

}  // namespace project