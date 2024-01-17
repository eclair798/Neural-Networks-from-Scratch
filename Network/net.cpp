#include "net.h"

namespace project {

namespace {

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

void SetActFuncs(FuncNames names, Net::ActFuncs& place) {
    for (const FuncName& name : names) {
        place.push_back(act_functions[name]);
    }
}

void SetDistFunc(FuncName name, DistanceFunction& place) {
    place = dist_functions[name];
}

}  // namespace

Net::Net(Sizes layer_sizes, FuncNames act_funcs) : layer_sizes_(layer_sizes) {
    SetActFuncs(act_funcs, act_funcs_);

    // TODO setting layers and random matrices
}

void Net::Train(const Data& data, double eps, size_t max_iter,
                /*Alg,*/ const FuncName& dist_func) {
    data_x_ = data.first;
    data_y_ = data.second;
    error_ = eps;
    max_iter_ = max_iter;

    SetDistFunc(dist_func, dist_func_);

    // TODO "training" part of Train
}

Vector Net::Run(Vector x) {
    for (Layer& layer : layers_) {
        x = layer.Calc(x);
    }
    return x;
}

void Net::Reset() {
    for (Layer& layer : layers_) {
        layer.Reset();
    }
    // TODO setting new random matrices on layers
}

}  // namespace project