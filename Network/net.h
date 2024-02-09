#pragma once

#include "loss_function.h"
#include "layer.h"

namespace project {

class Net {
    using Layers = std::vector<Layer>;
    using AFNames = std::vector<AFName>;

public:
    Net(Sizes layer_sizes, const AFNames& act_funcs);
    void Train(const Data& data, DataType eps, size_t max_iter,
               /*Alg,*/ const LFName& dist_func);  // TODO

    Vector Calc(const Vector& x) const;
    void ImproveAccuracy();
    DataType CalculateError();

private:
    Layers layers_;
    LossFunction dist_func_;
};

}  // namespace project
