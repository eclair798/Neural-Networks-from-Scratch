#pragma once

#include "activation_function.h"

namespace project {

ActivationFunction::ActivationFunction(Func calc, Func der_calc)
    : calc_(std::move(calc)), derivative_(std::move(der_calc)) {
}

DataType ActivationFunction::Calc(DataType xi) const {
    DataType result = calc_(xi);
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

DataType ActivationFunction::Derivative(DataType xi) const {
    DataType result = derivative_(xi);
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

Vector ActivationFunction::Calc(const Vector& x) const {
    Vector result = x.unaryExpr([this](DataType xi) { return Calc(xi); });
    assert(result.allFinite() && "Not finite data");
    return result;
}

Matrix ActivationFunction::Derivative(const Vector& x) const {
    Vector new_vector = x.unaryExpr([this](DataType xi) { return Derivative(xi); });
    assert(new_vector.allFinite() && "Not finite data");
    Matrix result = new_vector.asDiagonal();
    return result;
}

Matrix ActivationFunction::Calc(const Matrix& xs) const {
    Matrix result = xs.array().unaryExpr([this](DataType xi) { return Calc(xi); }).matrix();
    assert(result.allFinite() && "Not finite data");
    return result;
}

Matrix ActivationFunction::Derivative(const Matrix& xs) const {
    Matrix result(xs.rows(), xs.cols());
    for (int col = 0; col < xs.cols(); ++col) {
        result.col(col) = Calc(Vector(xs.col(col)));
    }
    assert(result.allFinite() && "Not finite data");
    return result;
}

namespace act_func_options {
DataType Sigmoid::Calc(DataType xi) {
    assert(std::isfinite(xi) && "Not finite data");
    DataType denominator = 1.0 + exp(-xi);
    assert(denominator != 0 && "Zero division");
    DataType result = 1.0 / denominator;
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

DataType Sigmoid::Derivative(DataType xi) {
    assert(std::isfinite(xi) && "Not finite data");
    DataType denominator = exp(xi) + exp(-xi) + 2.0;
    assert(denominator != 0 && "Zero division");
    DataType result = 1.0 / denominator;
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

DataType Tanh::Calc(DataType xi) {
    assert(std::isfinite(xi) && "Not finite data");
    DataType result = tanh(xi);
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

DataType Tanh::Derivative(DataType xi) {
    assert(std::isfinite(xi) && "Not finite data");
    DataType denominator = cosh(xi) * cosh(xi);
    assert(denominator != 0 && "Zero division");
    DataType result = 1.0 / denominator;
    assert(std::isfinite(result) && "Not finite data");
    return result;
}

DataType ReLU::Calc(DataType xi) {
    assert(std::isfinite(xi) && "Not finite data");
    return (xi > 0) ? xi : 0.0;
}

DataType ReLU::Derivative(DataType xi) {
    assert(std::isfinite(xi) && "Not finite data");
    return (xi > 0) ? 1.0 : 0.0;
}

DataType Linear::Calc(DataType xi) {
    assert(std::isfinite(xi) && "Not finite data");
    return xi;
}

DataType Linear::Derivative(DataType) {
    return 1.0;
}

Vector Softmax::Calc(const Vector& x) {
    assert(x.allFinite() && "Not finite data");
    Vector exp_x = (x.array() - x.maxCoeff()).exp();
    assert(exp_x.allFinite() && "Not finite data");

    DataType sum_exp_x = exp_x.sum();
    assert(sum_exp_x != 0 && "Zero division");

    Vector result = exp_x / sum_exp_x;
    assert(result.allFinite() && "Not finite data");
    return result;
}

Matrix Softmax::Derivative(const Vector& x) {
    assert(x.allFinite() && "Not finite data");
    Vector softmax_x = Calc(x);
    assert(softmax_x.allFinite() && "Not finite data");

    Matrix jacobian = softmax_x.asDiagonal();
    jacobian -= softmax_x * softmax_x.transpose();

    assert(jacobian.allFinite() && "Not finite data");
    return jacobian;
}

Matrix Softmax::Calc(const Matrix& xs) {
    Matrix result(xs.rows(), xs.cols());
    for (int col = 0; col < xs.cols(); ++col) {
        result.col(col) = Calc(Vector(xs.col(col)));
    }
    assert(result.allFinite() && "Not finite data");
    return result;
}

}  // namespace act_func_options

Sigma AFMake(AFName name) {
    switch (name) {
        case AFName::Sigmoid:
            return ActivationFunction(act_func_options::Sigmoid::Calc,
                                      act_func_options::Sigmoid::Derivative);
        case AFName::Tanh:
            return ActivationFunction(act_func_options::Tanh::Calc,
                                      act_func_options::Tanh::Derivative);
        case AFName::ReLU:
            return ActivationFunction(act_func_options::ReLU::Calc,
                                      act_func_options::ReLU::Derivative);
        case AFName::Linear:
            return ActivationFunction(act_func_options::Linear::Calc,
                                      act_func_options::Linear::Derivative);
        case AFName::Softmax:
            return act_func_options::Softmax();
        default:
            assert(false && "Unknown Activation function");
    }
}

}  // namespace project
