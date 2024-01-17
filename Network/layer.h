#pragma once

#include "activation_function.h"

namespace project {

class Layer {
public:
    using Index = size_t;

    Layer() = default;

    Layer(ActivationFunction func);

    void Reset();

    // x -> (ax + b) -> sigma(ax + b)
    Vector Calc(Vector x);  // TODO

    Matrix get_a_correction(); // TODO

    Vector get_b_correction(); // TODO

    RowVector PushU();  // TODO

private:
    Index layer_index_;

    //    Matrix matrix_u_;
    RowVector vector_u_;

    Matrix matrix_a_;
    Vector vector_b_;

    ActivationFunction sigma_;
};

}  // namespace project