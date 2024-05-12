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
    constexpr static DataType kNorm = 1.0 / 12.0;
    assert(kNorm != 0 && "Zero norm");
    Matrix result = Eigen::Rand::normal<Matrix>(rows, cols, GetRng()) * kNorm;
    assert(result.allFinite() && "Not finite data");
    return result;
}

Matrix CallCalc(const Sigma& sigma, const Matrix& xs) {
    Matrix lin_output;
    if (std::holds_alternative<ActivationFunction>(sigma)) {
        lin_output = std::get<ActivationFunction>(sigma).Calc(xs);
    } else {
        lin_output = std::get<act_func_options::Softmax>(sigma).Calc(xs);
    }
    return lin_output;
}

Matrix CallDerivative(const Sigma& sigma, const Vector& x) {
    Matrix der;
    if (std::holds_alternative<ActivationFunction>(sigma)) {
        der = std::get<ActivationFunction>(sigma).Derivative(x);
    } else {
        der = std::get<act_func_options::Softmax>(sigma).Derivative(x);
    }
    return der;
}

}  // namespace

Layer::Layer(Index input_size, Index output_size, Sigma func)
    : input_size_(input_size), output_size_(output_size), sigma_(std::move(func)) {
    assert(input_size != 0 && output_size != 0 && "The dimension of the layer cannot be zero");
}

Matrix Layer::Calc(const Matrix& xs) const {
    assert(xs.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(xs.allFinite() && "Not finite data");
    assert(matrix_a_.allFinite() && "Not finite data");
    assert(vector_b_.allFinite() && "Not finite data");
    Matrix lin_output = Linear(xs);
    assert(lin_output.allFinite() && "Not finite data");
    Matrix result = CallCalc(sigma_, lin_output);

    assert(result.allFinite() && "Not finite data");
    return result;
}

Vector Layer::Linear(const Vector& x) const {
    assert(x.allFinite() && "Not finite data");
    Vector lin_output = matrix_a_ * x + vector_b_;
    assert(lin_output.allFinite() && "Not finite data");
    return lin_output;
}

Matrix Layer::Linear(const Matrix& xs) const {
    assert(xs.allFinite() && "Not finite data");
    Matrix lin_output = (matrix_a_ * xs).colwise() + vector_b_;
    assert(lin_output.allFinite() && "Not finite data");
    return lin_output;
}

RowVector Layer::PushU(const RowVector& u, const Vector& x) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradient");
    assert(x.rows() == input_size_ && "Incorrect dimension of the input vector");
    assert(u.allFinite() && "Not finite data");
    assert(x.allFinite() && "Not finite data");

    Vector lin_output = Linear(x);

    Matrix der = CallDerivative(sigma_, lin_output);
    RowVector result = u * der * matrix_a_;

    assert(result.allFinite() && "Not finite data");
    return result;
}

Matrix Layer::PushU(const Matrix& us, const Matrix& xs) const {
    assert(us.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(xs.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(us.rows() == xs.cols() && "Incorrect count of vectors");

    assert(us.allFinite() && "Not finite data");
    assert(xs.allFinite() && "Not finite data");

    Matrix result(us.rows(), xs.rows());

    if (std::holds_alternative<ActivationFunction>(sigma_)) {
        Matrix lin_output = Linear(xs);
        Matrix der_batch = std::get<ActivationFunction>(sigma_).Derivative(lin_output);
        result = (der_batch.transpose().array() * us.array()).matrix() * matrix_a_;
        return result;
    }
    for (Index i = 0; i < result.rows(); ++i) {
        RowVector vec_u = us.row(i);
        Vector vec_x = xs.col(i);
        result.row(i) = PushU(vec_u, vec_x);
    }

    assert(result.allFinite() && "Not finite data");
    return result;
}

Matrix Layer::GetACorrection(const RowVector& u, const Vector& x) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradient");
    assert(x.rows() == input_size_ && "Incorrect dimension of the input vector");

    assert(u.allFinite() && "Not finite data");
    assert(x.allFinite() && "Not finite data");

    Vector lin_output = Linear(x);
    Matrix der = CallDerivative(sigma_, lin_output);
    Matrix result = der * u.transpose() * x.transpose();

    assert(result.allFinite() && "Not finite data");
    return result;
}

Matrix Layer::GetACorrection(const Matrix& us, const Matrix& xs) const {
    assert(us.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(xs.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(us.rows() == xs.cols() && "Incorrect count of vectors");

    assert(us.allFinite() && "Not finite data");
    assert(xs.allFinite() && "Not finite data");

    Matrix result(matrix_a_.rows(), matrix_a_.cols());

    if (std::holds_alternative<ActivationFunction>(sigma_)) {
        Matrix lin_output = Linear(xs);
        Matrix der_batch = std::get<ActivationFunction>(sigma_).Derivative(lin_output);
        result = (der_batch.array() * us.transpose().array()).matrix() * xs.transpose();
        result /= us.rows();
        return result;
    }

    for (Index i = 0; i < us.rows(); ++i) {
        RowVector vec_u = us.row(i);
        Vector vec_x = xs.col(i);
        result += GetACorrection(vec_u, vec_x);
    }
    result /= us.rows();

    assert(result.allFinite() && "Not finite data");
    return result;
}

Vector Layer::GetBCorrection(const RowVector& u, const Vector& x) const {
    assert(u.cols() == output_size_ && "Incorrect dimension of the gradient");
    assert(x.rows() == input_size_ && "Incorrect dimension of the input vector");
    assert(u.allFinite() && "Not finite data");
    assert(x.allFinite() && "Not finite data");
    Vector lin_output = Linear(x);
    Matrix der = CallDerivative(sigma_, lin_output);
    RowVector result = der * u.transpose();

    assert(result.allFinite() && "Not finite data");
    return result;
}

Vector Layer::GetBCorrection(const Matrix& us, const Matrix& xs) const {
    assert(us.cols() == output_size_ && "Incorrect dimension of the gradients");
    assert(xs.rows() == input_size_ && "Incorrect dimension of the input vectors");
    assert(us.rows() == xs.cols() && "Incorrect count of vectors");

    assert(us.allFinite() && "Not finite data");
    assert(xs.allFinite() && "Not finite data");

    Vector result(vector_b_.rows());

    if (std::holds_alternative<ActivationFunction>(sigma_)) {
        Matrix lin_output = Linear(xs);
        Matrix der_batch = std::get<ActivationFunction>(sigma_).Derivative(lin_output);
        result = (der_batch.array() * us.transpose().array()).rowwise().sum();
        result /= us.rows();
        return result;
    }

    for (Index i = 0; i < us.rows(); ++i) {
        RowVector vec_u = us.row(i);
        Vector vec_x = xs.col(i);
        result += GetBCorrection(vec_u, vec_x);
    }
    result /= us.rows();

    assert(result.allFinite() && "Not finite data");
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
