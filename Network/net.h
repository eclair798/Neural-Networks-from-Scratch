#pragma once

#include "activation_function.h"
#include "loss_function.h"

namespace project {

using ActFuncs = std::vector<ActivationFunction>;

class Net {
public:
    Net(const Sizes& layer_sizes, const FuncNames& act_funcs);

    void Train(const DataXY& data_xy, double eps, size_t max_iter,
               /*Alg,*/ const FuncName& dist_func);  // TODO

private:
    // TODO

    Sizes layer_sizes_;
    ActFuncs act_funcs_ = {};
    DistanceFunction dist_func_;

    double error_;
    size_t max_iter_;

    Data data_x_;
    Data data_y_;
};

namespace details {

extern std::map<FuncName, ActivationFunction> act_functions;

extern std::map<FuncName, DistanceFunction> dist_functions;

void SetActFuncs(FuncNames names, ActFuncs& place);

void SetDistFunc(FuncName name, DistanceFunction& place);

}  // namespace details

}  // namespace project
