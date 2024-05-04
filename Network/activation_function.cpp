#pragma once

#include "activation_function.h"

namespace project {

ActivationFunction::ActivationFunction(Func calc, FuncDerivative der_calc, bool diag_der,
                                       FuncDerivativeDim1 der_dim1_calc)
    : calc_(std::move(calc)),
      derivative_(std::move(der_calc)),
      is_diagonal_derivative_(diag_der),
      derivative_dim1_(std::move(der_dim1_calc)) {
}

ActivationFunction ActivationFunction::Make(AFName name) {
    switch (name) {
        case AFName::Sigmoid:
            return ActivationFunction(act_func_options::Sigmoid::Calc,
                                      act_func_options::Sigmoid::Derivative, true,
                                      act_func_options::Sigmoid::DerivativeDim1);
        case AFName::Tanh:
            return ActivationFunction(act_func_options::Tanh::Calc,
                                      act_func_options::Tanh::Derivative, true,
                                      act_func_options::Tanh::DerivativeDim1);
        case AFName::ReLU:
            return ActivationFunction(act_func_options::ReLU::Calc,
                                      act_func_options::ReLU::Derivative, true,
                                      act_func_options::ReLU::DerivativeDim1);
        case AFName::Linear:
            return ActivationFunction(act_func_options::Linear::Calc,
                                      act_func_options::Linear::Derivative, true,
                                      act_func_options::Linear::DerivativeDim1);
        case AFName::Softmax:
            return ActivationFunction(act_func_options::Softmax::Calc,
                                      act_func_options::Softmax::Derivative, false);
        default:
            assert(false && "Unknown Activation function");
    }
}

Vector ActivationFunction::Calc(const Vector& x) const {
    Vector result = calc_(x);
    return result;
}
Matrix ActivationFunction::Derivative(const Vector& x) const {
    Matrix result = derivative_(x);
    return result;
}
Matrix ActivationFunction::CalcBatch(const Matrix& xs) const {
    Matrix result(xs.rows(), xs.cols());
    for (int col = 0; col < xs.cols(); ++col) {
        result.col(col) = Calc(xs.col(col));
        assert(result.col(col).allFinite() && "Not finite data");
    }
    return result;
}

Matrix ActivationFunction::DerivativeBatch(const Matrix& x) const {
    assert(is_diagonal_derivative_ && "Taking batch derivative from non diagonal function");
    Matrix result = x.unaryExpr(derivative_dim1_);
    return result;
}
bool ActivationFunction::IsDiagonalDerivative() const {
    return is_diagonal_derivative_;
}

namespace act_func_options {

DataType Sigmoid::CalcDim1(DataType x) {
    return 1.0 / (1.0 + exp(-x));
}
DataType Sigmoid::DerivativeDim1(DataType x) {
    return 1.0 / (exp(x) + exp(-x) + 2.0);
}
Vector Sigmoid::Calc(const Vector& x) {
    return x.unaryExpr([](DataType x) { return CalcDim1(x); });
}
Matrix Sigmoid::Derivative(const Vector& x) {
    Vector result = x.unaryExpr([](DataType x) { return Sigmoid::DerivativeDim1(x); });
    return result.asDiagonal();
}

DataType Tanh::CalcDim1(DataType x) {
    return tanh(x);
}
DataType Tanh::DerivativeDim1(DataType x) {
    return 1.0 / (cosh(x) * cosh(x));
}
Vector Tanh::Calc(const Vector& x) {
    return x.unaryExpr([](DataType x) { return CalcDim1(x); });
}
Matrix Tanh::Derivative(const Vector& x) {
    Vector result = x.unaryExpr([](DataType x) { return Tanh::DerivativeDim1(x); });
    return result.asDiagonal();
}

DataType ReLU::CalcDim1(DataType x) {
    return (x > 0) ? x : 0.0;
}
DataType ReLU::DerivativeDim1(DataType x) {
    return (x > 0) ? 1.0 : 0.0;
}
Vector ReLU::Calc(const Vector& x) {
    assert(x.allFinite() && "Not finite data");
    Vector result = x.unaryExpr([](DataType x) { return CalcDim1(x); });
    assert(result.allFinite() && "Not finite data");
    return result;
}
Matrix ReLU::Derivative(const Vector& x) {
    assert(x.allFinite() && "Not finite data");
    Vector result = x.unaryExpr([](DataType x) { return ReLU::DerivativeDim1(x); });
    assert(result.allFinite() && "Not finite data");
    return result.asDiagonal();
}

DataType Linear::CalcDim1(DataType x) {
    return x;
}
DataType Linear::DerivativeDim1(DataType) {
    return 1.0;
}
Vector Linear::Calc(const Vector& x) {
    return x.unaryExpr([](DataType x) { return CalcDim1(x); });
}
Matrix Linear::Derivative(const Vector& x) {
    Vector result = x.unaryExpr([](DataType x) { return Linear::DerivativeDim1(x); });
    return result.asDiagonal();
}

Vector Softmax::Calc(const Vector& x) {
    assert(x.allFinite() && "Not finite data");
    Vector exp_x = (x.array() - x.maxCoeff()).exp();
    assert(exp_x.allFinite() && "Not finite data");
    DataType sum_exp_x = exp_x.sum();
    assert(sum_exp_x != 0 && "Zero division");
    return exp_x / sum_exp_x;
}
Matrix Softmax::Derivative(const Vector& x) {
    assert(x.allFinite() && "Not finite data");
    Vector softmax_x = Calc(x);
    assert(softmax_x.allFinite() && "Not finite data");

    Matrix jacobian = softmax_x.asDiagonal();
    jacobian -= softmax_x * softmax_x.transpose();
    return jacobian;
}

}  // namespace act_func_options

}  // namespace project
