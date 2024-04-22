#pragma once

#include "activation_function.h"

namespace project {

ActivationFunction::ActivationFunction(FuncAct calc, FuncDerivativeAct der_calc)
    : calc_(std::move(calc)), derivative_(std::move(der_calc)) {
}
ActivationFunction ActivationFunction::Make(AFName name) {
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
            return ActivationFunction(act_func_options::Softmax::Calc,
                                      act_func_options::Softmax::Derivative);
        default:
            assert(false && "Unknown Activation function");
    }
}

Vector ActivationFunction::Calc(const Vector& vector) const {
    Vector result = calc_(vector);
    return result;
}
Matrix ActivationFunction::Derivative(const Vector& vector) const {
    Matrix result = derivative_(vector);
    return result;
}
Matrix ActivationFunction::CalcMatrix(const Matrix& matrix) const {
    Matrix result(matrix.rows(), matrix.cols());
    for (int col = 0; col < matrix.cols(); ++col) {
        result.col(col) = Calc(matrix.col(col));
        assert(result.col(col).allFinite() && "Not finite matrix");
    }
    return result;
}

namespace act_func_options {

DataType Sigmoid::CalcDim1(DataType x) {
    return 1.0 / (1.0 + exp(-x));
}
DataType Sigmoid::DerivativeDim1(DataType x) {
    return 1.0 / (exp(x) + exp(-x) + 2.0);
}
Vector Sigmoid::Calc(const Vector& vector) {
    return vector.unaryExpr([](DataType x) { return CalcDim1(x); });
}
Matrix Sigmoid::Derivative(const Vector& vector) {
    Vector result = vector.unaryExpr([](DataType x) { return Sigmoid::DerivativeDim1(x); });
    return result.asDiagonal();
}

DataType Tanh::CalcDim1(DataType x) {
    return tanh(x);
}
DataType Tanh::DerivativeDim1(DataType x) {
    return 1.0 / (cosh(x) * cosh(x));
}
Vector Tanh::Calc(const Vector& vector) {
    return vector.unaryExpr([](DataType x) { return CalcDim1(x); });
}
Matrix Tanh::Derivative(const Vector& vector) {
    Vector result = vector.unaryExpr([](DataType x) { return Tanh::DerivativeDim1(x); });
    return result.asDiagonal();
}

DataType ReLU::CalcDim1(DataType x) {
    return (x > 0) ? x : 0.0;
}
DataType ReLU::DerivativeDim1(DataType x) {
    return (x > 0) ? 1.0 : 0.0;
}
Vector ReLU::Calc(const Vector& vector) {
    assert(vector.allFinite() && "Not finite vector");
    Vector result = vector.unaryExpr([](DataType x) { return CalcDim1(x); });
    assert(result.allFinite() && "Not finite vector");
    return result;
}
Matrix ReLU::Derivative(const Vector& vector) {
    assert(vector.allFinite() && "Not finite vector");
    Vector result = vector.unaryExpr([](DataType x) { return ReLU::DerivativeDim1(x); });
    assert(result.allFinite() && "Not finite vector");
    return result.asDiagonal();
}

DataType Linear::CalcDim1(DataType x) {
    return x;
}
DataType Linear::DerivativeDim1(DataType) {
    return 1.0;
}
Vector Linear::Calc(const Vector& vector) {
    return vector.unaryExpr([](DataType x) { return CalcDim1(x); });
}
Matrix Linear::Derivative(const Vector& vector) {
    Vector result = vector.unaryExpr([](DataType x) { return Linear::DerivativeDim1(x); });
    return result.asDiagonal();
}

Vector Softmax::Calc(const Vector& vector) {
    assert(vector.allFinite() && "Not finite vector");
    //    Vector exp_x = (vector.array()).exp();
    Vector exp_x = (vector.array() - vector.maxCoeff()).exp();
    assert(exp_x.allFinite() && "Not finite vector");
    DataType sum_exp_x = exp_x.sum();
    assert(sum_exp_x != 0 && "Zero division");
    return exp_x / sum_exp_x;
}
Matrix Softmax::Derivative(const Vector& vector) {
    assert(vector.allFinite() && "Not finite vector");
    Vector softmax_x = Calc(vector);
    assert(softmax_x.allFinite() && "Not finite vector");

    Matrix jacobian = softmax_x.asDiagonal();
    jacobian -= softmax_x * softmax_x.transpose();
    return jacobian;
}

}  // namespace act_func_options

}  // namespace project
