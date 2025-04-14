#pragma once

#include "activation_function.h"

namespace nn {
class Layer {
public:
    Layer() = default;
    Layer(Index input_size, Index output_size, Sigma func);
    Matrix Calc(const Matrix& xs) const;

    Vector Linear(const Vector& x) const;
    Matrix Linear(const Matrix& xs) const;

    RowVector PushU(const RowVector& u, const Vector& x) const;
    Matrix PushU(const Matrix& us, const Matrix& xs) const;

    Matrix GetACorrection(const RowVector& u, const Vector& x) const;
    Matrix GetACorrection(const Matrix& us, const Matrix& xs) const;

    Vector GetBCorrection(const RowVector& u, const Vector& x) const;
    Vector GetBCorrection(const Matrix& us, const Matrix& xs) const;

    void SetParam(Matrix&& matrix_a, Vector&& vector_b);

    void SetRandParam();

    const Matrix& GetA() const;
    const Vector& GetB() const;

    void CorrectA(const Matrix& delta, DataType learning_rate);
    void CorrectB(const Vector& delta, DataType learning_rate);

    Index GetInputSize() const;
    Index GetOutputSize() const;

private:
    Index input_size_;
    Index output_size_;
    Matrix matrix_a_;  // outputS * inputS
    Vector vector_b_;  // outputS * 1
    std::variant<ActivationFunction, act_func_options::Softmax> sigma_;
};

}  // namespace nn
