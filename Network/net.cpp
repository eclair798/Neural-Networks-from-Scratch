#include "net.h"

project::Net::Net(const Sizes& layer_sizes, const FuncNames& act_funcs)
    : layer_sizes_(layer_sizes) {
    details::SetActFuncs(act_funcs, act_funcs_);
}

void project::Net::Train(const DataXY& data_xy, double eps, size_t max_iter,
                         /*Alg,*/ const FuncName& dist_func) {
    data_x_ = data_xy.first;
    data_y_ = data_xy.second;
    error_ = eps;
    max_iter_ = max_iter;

    details::SetDistFunc(dist_func, dist_func_);

    // TODO
}

namespace project::details {

std::map<FuncName, ActivationFunction> act_functions = {
    {"Sigmoid",
     ActivationFunction(act_func_options::Sigmoid::Calc, act_func_options::Sigmoid::Derivative)},
    {"Tanh", ActivationFunction(act_func_options::Tanh::Calc, act_func_options::Tanh::Derivative)},
    {"ReLU", ActivationFunction(act_func_options::ReLU::Calc, act_func_options::ReLU::Derivative)},
    {"Linear",
     ActivationFunction(act_func_options::Linear::Calc, act_func_options::Linear::Derivative)}};

std::map<FuncName, DistanceFunction> dist_functions = {
    {"SquaredEuclidean", DistanceFunction(dist_func_options::SquaredEuclidean::Dist,
                                          dist_func_options::SquaredEuclidean::DerivativeDist)},
    {"Manhattan", DistanceFunction(dist_func_options::Manhattan::Dist,
                                   dist_func_options::Manhattan::DerivativeDist)}};

}  // namespace project::details

void project::details::SetActFuncs(FuncNames names, ActFuncs& place) {
    for (const FuncName& name : names) {
        place.push_back(act_functions[name]);
    }
}

void project::details::SetDistFunc(FuncName name, DistanceFunction& place) {
    place = dist_functions[name];
}
