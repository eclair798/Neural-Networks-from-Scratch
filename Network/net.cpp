#pragma once

#include "net.h"

namespace project {

Net::Net(Sizes layer_sizes, const AFNames& act_funcs) {
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        layers_.emplace_back(layer_sizes[i], layer_sizes[i + 1],
                             ActivationFunction::Make(act_funcs[i]));
    }
}
void Net::Train(const Data& data, DataType eps, size_t max_iter,
                /*Alg,*/ const LFName& dist_func) {
    dist_func_ = LossFunction::Make(dist_func);
    // TODO "training" part of Train
}

Vector Net::Calc(const Vector& x) const {
    Vector cur_x = x;
    for (const Layer& layer : layers_) {
        cur_x = layer.Calc(cur_x);
    }
    return x;
}

// void Net::ImproveAccuracy() {
// }
//
// DataType Net::CalculateError() {
//     return 0;
// }

}  // namespace project
