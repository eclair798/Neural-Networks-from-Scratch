#pragma once

#include "activation_function.h"

namespace project {
class Layer {
public:
    Layer() = default;
    Layer(Index input_size, Index output_size, ActivationFunction func);
    // x -> (ax + b) -> sigma(ax + b)
    Vector Calc(const Vector& x) const;  // TODO

    RowVector PushU(const RowVector& u, const Vector& input);
    Matrix PushU(const Matrix& u, const Matrix& input);

    Matrix GetACorrection(const RowVector& u, const Vector& input);
    Matrix GetACorrection(const Matrix& u, const Matrix& input);

    Vector GetBCorrection(const RowVector& u, const Vector& input);
    Matrix GetBCorrection(const Matrix& u, const Matrix& input);

    void SetNewParams();  // TODO

private:
    Index input_size_;
    Index output_size_;
    Matrix matrix_a_;  // outputS * inputS
    Vector vector_b_;  // outputS * 1
    ActivationFunction sigma_;
};
}  // namespace project
