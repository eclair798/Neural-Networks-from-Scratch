#pragma once

#include "activation_function.h"
#include "loss_function.h"

namespace project {

using Sizes = std::initializer_list<size_t>;

using DataVec = std::initializer_list<std::initializer_list<double>>;
using Data = std::pair<DataVec, DataVec>;

using FuncName = std::string;
using FuncNames = std::initializer_list<FuncName>;
using ActFuncs = std::vector<std::shared_ptr<ActivationFunction>>;

namespace details {

extern std::map<FuncName, std::shared_ptr<ActivationFunction>> act_functions;

extern std::map<FuncName, std::shared_ptr<DistanceFunction>> dist_functions;

void SetActFuncs(FuncNames names, ActFuncs& place);

void SetDistFunc(FuncName name, std::shared_ptr<DistanceFunction>& place);

void CreateMatrix(const project::DataVec& data, std::shared_ptr<Eigen::MatrixXd>& place);

}  // namespace details

}  // namespace project