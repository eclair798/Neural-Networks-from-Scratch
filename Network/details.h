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

std::map<FuncName, std::shared_ptr<ActivationFunction>> act_functions = {
    {"Sigmoid", std::shared_ptr<ActivationFunction>(new act_func_options::Sigmoid())},
    {"Tanh", std::shared_ptr<ActivationFunction>(new act_func_options::Tanh())},
    {"ReLU", std::shared_ptr<ActivationFunction>(new act_func_options::ReLU())},
    {"Linear", std::shared_ptr<ActivationFunction>(new act_func_options::Linear())}};

std::map<FuncName, std::shared_ptr<DistanceFunction>> dist_functions = {
    {"SquaredEuclidean",
     std::shared_ptr<DistanceFunction>(new dist_func_options::SquaredEuclidean())},
    {"Manhattan", std::shared_ptr<DistanceFunction>(new dist_func_options::Manhattan())}};

void SetActFuncs(FuncNames names, ActFuncs& place);

void SetDistFunc(FuncName name, std::shared_ptr<DistanceFunction>& place);

void CreateMatrix(const project::DataVec& data, std::shared_ptr<Eigen::MatrixXd>& place);

}  // namespace details

}  // namespace project