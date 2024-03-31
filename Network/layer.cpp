#pragma once

#include "layer.h"

namespace project {

namespace {
using RandGen = Eigen::Rand::Vmt19937_64;
RandGen& GetRng() {
    static RandGen rng = 1;
    return rng;
}
Matrix GenerateRandomNormalMatrix(Index rows, Index cols) {
    return Eigen::Rand::normal<Matrix>(rows, cols, GetRng());
}
}  // namespace

Layer::Layer(Index input_size, Index output_size, ActivationFunction func)
    : input_size_(input_size), output_size_(output_size), sigma_(std::move(func)) {
    assert(input_size != 0 && output_size != 0 && "The dimension of the layer cannot be zero");
    SetNewParams();
}

Vector Layer::Calc(const Vector& input) const {
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vector");
    Vector lin_output = matrix_a_ * input + vector_b_;
    return sigma_.Calc(lin_output);
}
Matrix Layer::Calc(const Matrix& input) const {
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    Matrix lin_output = (matrix_a_ * input).colwise() + vector_b_;
    return sigma_.Calc(lin_output);
}

RowVector Layer::PushU(const RowVector& u, const Vector& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradient");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vector");
    Vector lin_output = matrix_a_ * input + vector_b_;
    return u * sigma_.Derivative(lin_output) * matrix_a_;
}
Matrix Layer::PushU(const Matrix& u, const Matrix& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    Matrix result(u.rows(), input.size());
    for (Index i = 0; i < result.rows(); ++i) {
        RowVector vec_u = u.row(i);
        Vector vec_input = input.col(i);
        result.row(i) = PushU(vec_u, vec_input);
    }
    return result;
}

Matrix Layer::GetACorrection(const RowVector& u, const Vector& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradient");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vector");
    Vector lin_output = matrix_a_ * input + vector_b_;
    return sigma_.Derivative(lin_output) * u.transpose() * input;
}
Matrix Layer::GetACorrection(const Matrix& u, const Matrix& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    Matrix result(matrix_a_.rows(), matrix_a_.cols());
    for (Index i = 0; i < u.rows(); ++i) {
        RowVector vec_u = u.row(i);
        Vector vec_input = input.col(i);
        result += GetACorrection(vec_u, vec_input);
    }
    result /= u.rows();
    return result;
}

Vector Layer::GetBCorrection(const RowVector& u, const Vector& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradient");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vector");
    return sigma_.Derivative(matrix_a_ * input + vector_b_) * u.transpose();
}
Matrix Layer::GetBCorrection(const Matrix& u, const Matrix& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    Vector result(vector_b_.rows());
    for (Index i = 0; i < u.rows(); ++i) {
        RowVector vec_u = u.row(i);
        Vector vec_input = input.col(i);
        result += GetBCorrection(vec_u, vec_input);
    }
    result /= u.rows();
    return result;
}

void Layer::SetNewParams() {
    matrix_a_ = GenerateRandomNormalMatrix(output_size_, input_size_);
    vector_b_ = GenerateRandomNormalMatrix(output_size_, 1);
}

void Layer::CorrectA(const Matrix& delta, DataType learning_rate) {
    assert(matrix_a_.cols() == delta.cols() && matrix_a_.rows() == delta.rows() && "Incorrect dimension of delta");
    matrix_a_ = matrix_a_ - delta * learning_rate;
}
void Layer::CorrectB(const Vector& delta, DataType learning_rate) {
    assert(vector_b_.cols() == delta.cols() && vector_b_.rows() == delta.rows() && "Incorrect dimension of delta");
    vector_b_ = vector_b_ - delta * learning_rate;
}

Index Layer::GetInputSize() const {
    return input_size_;
}
Index Layer::GetOutputSize() const {
    return output_size_;
}

}  // namespace project
