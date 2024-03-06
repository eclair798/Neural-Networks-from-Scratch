#pragma once

#include "layer.h"

namespace project {

namespace {

using NormalDistribution = Eigen::Rand::NormalGen<DataType>;

Matrix GenerateRandomNormalMatrix(Index rows, Index cols) {
    NormalDistribution normal_distribution(0.0, 1.0);
    Matrix random_matrix(rows, cols);
    // TODO generator
}

}  // namespace

Vector Layer::Calc(const Vector& x) const {
    return sigma_.Calc(matrix_a_ * x + vector_b_);
}
Layer::Layer(Index input_size, Index output_size, ActivationFunction func)
    : input_size_(input_size), output_size_(output_size), sigma_(std::move(func)) {
    SetNewParams();
}
void Layer::SetNewParams() {
    // TODO set new random matrices
}

RowVector Layer::PushU(const RowVector& u, const Vector& input) {
    return u * sigma_.Derivative(matrix_a_ * input + vector_b_) * matrix_a_;
}
Matrix Layer::PushU(const Matrix& u, const Matrix& input) {
    return project::Matrix();
}


Matrix Layer::GetACorrection(const RowVector& u, const Vector& input) {
    return sigma_.Derivative(matrix_a_ * input + vector_b_) * u.transpose() * matrix_a_;
}
Vector Layer::GetBCorrection(const RowVector& u, const Vector& input) {
    return sigma_.Derivative(matrix_a_ * input + vector_b_) * u.transpose();
}

}  // namespace project
