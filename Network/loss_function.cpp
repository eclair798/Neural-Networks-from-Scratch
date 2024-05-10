#pragma once

#include "loss_function.h"

namespace project {

LossFunction::LossFunction(FuncDist calc, FuncGrad der_calc)
    : dist_(std::move(calc)), grad_(std::move(der_calc)) {
}

LossFunction LossFunction::Make(LFName name) {
    switch (name) {
        case LFName::MSE:
            return LossFunction(loss_func_options::MSE::Dist, loss_func_options::MSE::Grad);
        case LFName::Manhattan:
            return LossFunction(loss_func_options::Manhattan::Dist,
                                loss_func_options::Manhattan::Grad);
        case LFName::CrossEntropy:
            return LossFunction(loss_func_options::CrossEntropy::Dist,
                                loss_func_options::CrossEntropy::Grad);
        default:
            assert(false && "Unknown Activation function");
    }
}

DataType LossFunction::Dist(const Vector& x, const Vector& y) const {
    assert(x.rows() == y.rows() &&
           "The distance between vectors of different dimensions cannot be considered");
    return dist_(x, y);
}

Vector LossFunction::Grad(const Vector& x, const Vector& y) const {
    assert(x.rows() == y.rows() &&
           "The distance between vectors of different dimensions cannot be considered");
    assert(x.allFinite() && "Not finite data");
    assert(y.allFinite() && "Not finite data");

    return grad_(x, y);
}

DataType LossFunction::Dist(const Matrix& xs, const Matrix& ys) const {
    assert(xs.rows() == ys.rows() &&
           "The distance between vectors of different dimensions cannot be considered");
    assert(xs.cols() == ys.cols() && "The number of vectors differs");

    assert(xs.allFinite() && "Not finite data");
    assert(ys.allFinite() && "Not finite data");

    DataType distance = 0.0;
    Index size = xs.cols();
    for (Index col_i = 0; col_i < size; ++col_i) {
        Vector x_i = xs.col(col_i);
        Vector y_i = ys.col(col_i);
        distance += Dist(x_i, y_i);
    }
    distance /= size;
    return distance;
}

Matrix LossFunction::Grad(const Matrix& xs, const Matrix& ys) const {
    assert(xs.rows() == ys.rows() &&
           "The distance between vectors of different dimensions cannot be considered");
    assert(xs.cols() == ys.cols() && "The number of vectors differs");

    assert(xs.allFinite() && "Not finite data");
    assert(ys.allFinite() && "Not finite data");

    Matrix matrix_u(xs.cols(), xs.rows());
    Index size = xs.cols();
    for (Index col_i = 0; col_i < size; ++col_i) {
        Vector x_i = xs.col(col_i);
        Vector y_i = ys.col(col_i);

        assert(x_i.allFinite() && "Not finite data");
        assert(y_i.allFinite() && "Not finite data");
        matrix_u.row(col_i) = Grad(x_i, y_i);
    }
    return matrix_u;
}

namespace loss_func_options {
DataType MSE::Dist(const Vector& x, const Vector& y) {
    DataType result = sqrt((x - y).norm());
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

Vector MSE::Grad(const Vector& x, const Vector& y) {
    Vector result = 2 * (x - y);
    assert(result.allFinite() && "Not finite data");
    return result;
}

DataType Manhattan::Dist(const Vector& x, const Vector& y) {
    DataType result = (x - y).lpNorm<1>();
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

Vector Manhattan::Grad(const Vector& x, const Vector& y) {
    Vector result = (x - y).unaryExpr([](DataType v) { return v > 0 ? 1.0 : -1.0; });
    assert(result.allFinite() && "Not finite data");
    return result;
}

DataType CrossEntropy::Dist(const Vector& x, const Vector& y) {
    DataType result = -(y.array() * x.array().log()).sum();
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

Vector CrossEntropy::Grad(const Vector& x, const Vector& y) {
    const double epsilon = 1e-7;
    Vector result = -(y.array() / (x.array() + epsilon));
    assert(result.allFinite() && "Not finite data");
    return result;
}

}  // namespace loss_func_options
}  // namespace project
