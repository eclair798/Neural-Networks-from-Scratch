#pragma once

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

#include <initializer_list>
#include <string>
#include <vector>
#include <memory>

namespace project {

using Sizes = std::initializer_list<size_t>;

using Data = Eigen::MatrixXd;
using DataXY = std::pair<Data, Data>;

using FuncName = std::string;
using FuncNames = std::initializer_list<FuncName>;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

using FuncAct = std::function<double(double)>;
using FuncDist = std::function<int(RowVector, RowVector)>;
using FuncDerivativeDist = std::function<RowVector(RowVector, RowVector)>;

}  // namespace project