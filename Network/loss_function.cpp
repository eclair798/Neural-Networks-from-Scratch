#pragma once

#include "loss_function.h"

namespace project {

LossFunction::LossFunction(FuncDist calc, FuncGrad der_calc)
    : dist_(std::move(calc)), grad_(std::move(der_calc)) {
}
LossFunction LossFunction::Make(LFName name) {
    switch (name) {
        case LFName::SquaredEuclidean:
            return LossFunction(loss_func_options::SquaredEuclidean::Dist,
                                loss_func_options::SquaredEuclidean::Grad);
        case LFName::Manhattan:
            return LossFunction(loss_func_options::Manhattan::Dist,
                                loss_func_options::Manhattan::Grad);
    }
}

DataType LossFunction::Dist(const Vector& x, const Vector& y) const {
    return dist_(x, y);
}
Vector LossFunction::Grad(const Vector& x, const Vector& y) const {
    return grad_(x, y);
}
DataType LossFunction::Dist(const Matrix& x, const Matrix& y) const {
    DataType distance = 0.0;
    Index size = x.cols();
    for (Index col_i = 0; col_i < size; ++col_i) {
        Vector x_i = x.col(col_i);
        Vector y_i = y.col(col_i);
        distance += Dist(x_i, y_i);
    }
    distance /= size;
    return distance;
}
Matrix LossFunction::Grad(const Matrix& x, const Matrix& y) const {
    Matrix matrix_u(x.cols(), x.rows());
    Index size = x.cols();
    for (Index col_i = 0; col_i < size; ++col_i) {
        Vector x_i = x.col(col_i);
        Vector y_i = y.col(col_i);
        matrix_u.row(col_i) = Grad(x_i, y_i);
    }
    return matrix_u;
}

namespace loss_func_options {
DataType SquaredEuclidean::Dist(const Vector& x, const Vector& y) {
    return sqrt((x - y).norm());
}
Vector SquaredEuclidean::Grad(const Vector& x, const Vector& y) {
    Vector vector_u = 2 * (x - y);
    return vector_u;
}

DataType Manhattan::Dist(const Vector& x, const Vector& y) {
    return (x - y).lpNorm<1>();
}
Vector Manhattan::Grad(const Vector& x, const Vector& y) {
    Vector vector_u = (x - y).unaryExpr([](DataType v) { return v > 0 ? 1.0 : -1.0; });
    return vector_u;
}
}  // namespace loss_func_options
}  // namespace project
