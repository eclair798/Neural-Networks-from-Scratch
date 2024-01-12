#pragma once

#include <vector>

#include "activation_function.h"

namespace project {

    using Sizes = std::vector<size_t>;
    using Data = std::pair<std::vector<double>, std::vector<double>>;


class Net {
public:
    Net(const Sizes& layer_sizes, const ActivationFunction* function); // TODO

    void Train(const Data& data, double eps, size_t MaxIter /*, Alg, phi*/); // TODO

private:
    // TODO
};

}
