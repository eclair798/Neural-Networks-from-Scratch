#pragma once

#include <initializer_list>
#include <string>
#include <vector>
#include <memory>

#include "activation_function.h"
#include "loss_function.h"

namespace project {

using Sizes = std::initializer_list<size_t>;

using DataVec = std::initializer_list<std::initializer_list<double>>;
using Data = std::pair<DataVec, DataVec>;

using FuncName = std::string;
using FuncNames = std::initializer_list<FuncName>;
using ActFuncs = std::vector<ActivationFunction>;

class Net {
public:
    Net(const Sizes& layer_sizes, const FuncNames act_funcs);  // TODO

    void Train(const Data& data, double eps, size_t max_iter, /*Alg,*/ FuncName dist_func);  // TODO

private:
    // TODO

    Sizes layer_sizes_;
    ActFuncs act_funcs_;
    DistanceFunction dist_func_;

    double error_;
    size_t max_iter_;

    std::unique_ptr<Eigen::MatrixXd> data_x_ = nullptr;
    std::unique_ptr<Eigen::MatrixXd> data_y_ = nullptr;
};

namespace details {

std::map<FuncName, ActivationFunction> act_functions = {{"Sigmoid", act_func_options::Sigmoid()},
                                                        {"Tanh", act_func_options::Tanh()},
                                                        {"ReLU", act_func_options::ReLU()},
                                                        {"Linear", act_func_options::Linear()}};
std::map<FuncName, DistanceFunction> dist_functions = {
    {"SquaredEuclidean", dist_func_options::SquaredEuclidean()},
    {"Manhattan", dist_func_options::Manhattan()}};

void SetActFuncs(FuncNames names, ActFuncs* place);

void SetDistFunc(FuncName name, DistanceFunction* place);

void CreateMatrix(const project::DataVec& data, std::unique_ptr<Eigen::MatrixXd>& place);

}  // namespace details

}  // namespace project
