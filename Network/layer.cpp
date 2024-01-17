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

Vector Layer::Calc(Vector x) {
    return sigma_.Calc(matrix_a_ * x + vector_b_);
}

Layer::Layer(ActivationFunction func) : sigma_(func) {
    // TODO generating random matrix_a & vector_b
}

void Layer::Reset() {
    vector_u_.setZero();

}

}  // namespace project