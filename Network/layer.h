#pragma once

#include "activation_function.h"

namespace project {
class Layer {
public:
    Layer() = default;
    Layer(Index input_size, Index output_size, ActivationFunction func);
    // x -> (ax + b) -> sigma(ax + b)
    Vector Calc(const Vector& input) const;
    Matrix Calc(const Matrix& input) const;

    RowVector PushU(const RowVector& u, const Vector& input) const;
    Matrix PushU(const Matrix& u, const Matrix& input) const;

    Matrix GetACorrection(const RowVector& u, const Vector& input) const;
    Matrix GetACorrection(const Matrix& u, const Matrix& input) const;

    Vector GetBCorrection(const RowVector& u, const Vector& input) const;
    Vector GetBCorrection(const Matrix& u, const Matrix& input) const;

    void SetParam(Matrix&& matrix_a, Vector&& vector_b);

    void SetRandParam();

    Matrix GetA();
    Vector GetB();

    void CorrectA(const Matrix& delta, DataType learning_rate);
    void CorrectB(const Vector& delta, DataType learning_rate);

    Index GetInputSize() const;
    Index GetOutputSize() const;

private:
    Index input_size_;
    Index output_size_;
    Matrix matrix_a_;  // outputS * inputS
    Vector vector_b_;  // outputS * 1
    ActivationFunction sigma_;
};
}  // namespace project
