#pragma once

#include "layer.h"

namespace project {

namespace {
using RandGen = Eigen::Rand::Vmt19937_64;
RandGen& GetRng() {
    static RandGen rng = 1;
    return rng;
}
Matrix GenerateRandomNormalMatrix(Index rows, Index cols, DataType norm = 1.0 / 12.0) {
    assert(norm != 0 && "Zero norm");
    Matrix result = Eigen::Rand::normal<Matrix>(rows, cols, GetRng()) * norm;
    assert(result.allFinite() && "Not finite data");
    return result;
}
}  // namespace

Layer::Layer(Index input_size, Index output_size, ActivationFunction func)
    : input_size_(input_size), output_size_(output_size), sigma_(std::move(func)) {
    assert(input_size != 0 && output_size != 0 && "The dimension of the layer cannot be zero");
}

Vector Layer::Calc(const Vector& input) const {
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vector");
    Vector lin_output = matrix_a_ * input + vector_b_;
    lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
    return sigma_.Calc(lin_output);
}
Matrix Layer::Calc(const Matrix& input) const {
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(input.allFinite() && "Not finite data");
    assert(matrix_a_.allFinite() && "Not finite data");
    assert(vector_b_.allFinite() && "Not finite data");
    Matrix lin_output = (matrix_a_ * input).colwise() + vector_b_;
    lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
    assert(lin_output.allFinite() && "Not finite data");
    Matrix result = sigma_.CalcBatch(lin_output);
    assert(result.allFinite() && "Not finite data");

    return result;
}

RowVector Layer::PushU(const RowVector& u, const Vector& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradient");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vector");
    assert(u.allFinite() && "Not finite data");
    assert(input.allFinite() && "Not finite data");

    Vector lin_output = matrix_a_ * input + vector_b_;
    lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });

    RowVector result = u * sigma_.Derivative(lin_output) * matrix_a_;
    return result;
}

Matrix Layer::PushU(const Matrix& u, const Matrix& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(u.rows() == input.cols() && "Incorrect count of vectors");

    assert(u.allFinite() && "Not finite data");
    assert(input.allFinite() && "Not finite data");

    Matrix result(u.rows(), input.rows());

    if (sigma_.IsDiagonalDerivative()) {
        Matrix lin_output = (matrix_a_ * input).colwise() + vector_b_;
        lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
        Matrix der_batch = sigma_.DerivativeBatch(lin_output);
        result = (der_batch.transpose().array() * u.array()).matrix() * matrix_a_;
        return result;
    }
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

    assert(u.allFinite() && "Not finite data");
    assert(input.allFinite() && "Not finite data");

    Vector lin_output = matrix_a_ * input + vector_b_;
    lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });

    Matrix der = sigma_.Derivative(lin_output);
    Matrix result = der * u.transpose() * input.transpose();
    return result;
}

Matrix Layer::GetACorrection(const Matrix& u, const Matrix& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(u.rows() == input.cols() && "Incorrect count of vectors");

    assert(u.allFinite() && "Not finite data");
    assert(input.allFinite() && "Not finite data");

    Matrix result(matrix_a_.rows(), matrix_a_.cols());

    if (sigma_.IsDiagonalDerivative()) {
        Matrix lin_output = (matrix_a_ * input).colwise() + vector_b_;
        lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
        Matrix der_batch = sigma_.DerivativeBatch(lin_output);
        //        result = der_batch.transpose().array() * (u.transpose() *
        //        input.transpose()).array();
        result = (der_batch.array() * u.transpose().array()).matrix() * input.transpose();
        result /= u.rows();
        return result;
    }

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

    assert(u.allFinite() && "Not finite data");
    assert(input.allFinite() && "Not finite data");

    Vector lin_output = matrix_a_ * input + vector_b_;
    lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });

    return sigma_.Derivative(lin_output) * u.transpose();
}
Vector Layer::GetBCorrection(const Matrix& u, const Matrix& input) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(input.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(u.rows() == input.cols() && "Incorrect count of vectors");

    assert(u.allFinite() && "Not finite data");
    assert(input.allFinite() && "Not finite data");

    Vector result(vector_b_.rows());

    if (sigma_.IsDiagonalDerivative()) {
        Matrix lin_output = (matrix_a_ * input).colwise() + vector_b_;
        lin_output = lin_output.unaryExpr([](double x) { return std::isfinite(x) ? x : 0.0; });
        Matrix der_batch = sigma_.DerivativeBatch(lin_output);
        result = (der_batch.array() * u.transpose().array()).rowwise().sum();
        result /= u.rows();
        return result;
    }

    for (Index i = 0; i < u.rows(); ++i) {
        RowVector vec_u = u.row(i);
        Vector vec_input = input.col(i);
        result += GetBCorrection(vec_u, vec_input);
    }
    result /= u.rows();
    return result;
}

void Layer::SetParam(Matrix&& matrix_a, Vector&& vector_b) {
    assert(matrix_a.rows() == output_size_ && matrix_a.cols() == input_size_ &&
           "Incorrect matrix size");
    assert(vector_b.rows() == output_size_ && "Incorrect vector size");

    assert(matrix_a.allFinite() && "Not finite data");
    assert(vector_b.allFinite() && "Not finite data");

    matrix_a_ = std::move(matrix_a);
    vector_b_ = std::move(vector_b);
}

void Layer::SetRandParam() {
    matrix_a_ = GenerateRandomNormalMatrix(output_size_, input_size_);
    vector_b_ = GenerateRandomNormalMatrix(output_size_, 1);
}

const Matrix& Layer::GetA() const {
    return matrix_a_;
}
const Vector& Layer::GetB() const {
    return vector_b_;
}

void Layer::CorrectA(const Matrix& delta, DataType learning_rate) {
    assert(matrix_a_.cols() == delta.cols() && matrix_a_.rows() == delta.rows() &&
           "Incorrect dimension of delta");
    matrix_a_ = matrix_a_ - delta * learning_rate;
}
void Layer::CorrectB(const Vector& delta, DataType learning_rate) {
    assert(vector_b_.cols() == delta.cols() && vector_b_.rows() == delta.rows() &&
           "Incorrect dimension of delta");
    vector_b_ = vector_b_ - delta * learning_rate;
}

Index Layer::GetInputSize() const {
    return input_size_;
}
Index Layer::GetOutputSize() const {
    return output_size_;
}

}  // namespace project
