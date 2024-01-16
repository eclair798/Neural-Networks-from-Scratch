#pragma once

#include "details.h"

namespace project {

class Layer {
    using Index = size_t;

    Layer() = default;

    // x -> (Ax + b) -> σ(Ax + b)
    DataType Calc(DataType x);  // TODO

    RowVector PushU(); // TODO



private:
    Index layer_index_;

    Matrix A_;
    Vector b_;
};

}  // namespace project