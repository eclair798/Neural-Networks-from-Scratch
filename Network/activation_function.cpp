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
    }
}

DataType ActivationFunction::Calc(DataType x) const {
    return calc_(x);
}
DataType ActivationFunction::Derivative(DataType x) const {
    return derivative_(x);
}
Vector ActivationFunction::Calc(const Vector& vector) const {
    Vector new_vector = vector.unaryExpr([this](DataType x) { return Calc(x); });
    return new_vector;
}
Matrix ActivationFunction::Derivative(const Vector& vector) const {
    Vector new_vector = vector.unaryExpr([this](DataType x) { return Derivative(x); });
    return new_vector.asDiagonal();
}

namespace act_func_options {
DataType Sigmoid::Calc(DataType x) {
    return 1.0 / (1.0 + exp(-x));
}
DataType Sigmoid::Derivative(DataType x) {
    //    return exp(x) / (exp(2.0 * x) + 2.0 * exp(x) + 1.0);
    return 1.0 / (exp(x) + exp(-x) + 2.0);
}

DataType Tanh::Calc(DataType x) {
    return tanh(x);
}
DataType Tanh::Derivative(DataType x) {
    return 1.0 / (cosh(x) * cosh(x));
}

DataType ReLU::Calc(DataType x) {
    return (x > 0) ? x : 0;
}
DataType ReLU::Derivative(DataType x) {
    return (x > 0) ? 1.0 : 0;
}

DataType Linear::Calc(DataType x) {
    return x;
}
DataType Linear::Derivative(DataType) {
    return 1.0;
}
}  // namespace act_func_options

}  // namespace project
