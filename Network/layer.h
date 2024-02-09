#pragma once

#include "activation_function.h"

namespace project {
class Layer {
public:
    Layer() = default;
    Layer(Index input_size, Index output_size, ActivationFunction func);

    void Reset();  //

    // x -> (ax + b) -> sigma(ax + b)
    Vector Calc(const Vector& x) const;  // TODO

    Matrix get_a_correction();         // TODO
    Vector get_b_correction();         // TODO
    RowVector PushU(const Vector& u);  // TODO

private:
    Index input_size_;
    Index output_size_;
    Matrix matrix_a_;  // outputS * inputS
    Vector vector_b_;  // outputS * 1
    ActivationFunction sigma_;
};
}  // namespace project
