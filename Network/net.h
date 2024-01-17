#pragma once

#include "loss_function.h"
#include "layer.h"

namespace project {

class Net {
public:
    using ActFuncs = std::vector<ActivationFunction>;
    using Layers = std::vector<Layer>;

    Net(Sizes layer_sizes, FuncNames act_funcs);

    void Train(const Data& data, double eps, size_t max_iter,
               /*Alg,*/ const FuncName& dist_func);  // TODO

    Vector Run(const Vector x);

    void Reset();

private:
    // TODO

    Sizes layer_sizes_; // from input_size to output_size
    Layers layers_;

    ActFuncs act_funcs_ = {};
    DistanceFunction dist_func_;

    DataType error_;
    size_t max_iter_;

    Matrix data_x_;
    Matrix data_y_;

    Index input_size_;
    Index output_size_;
};

namespace details {}  // namespace details

}  // namespace project
