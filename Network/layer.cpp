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
    // TODO generating random matrix_a & vector_b
}
void Layer::Reset() {
    // TODO set new random matrices
}

}  // namespace project
