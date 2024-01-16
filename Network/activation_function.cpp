#pragma once

#include "activation_function.h"

project::DataType project::ActivationFunction::Calc(DataType x) {
    return calc_(x);
}

project::DataType project::ActivationFunction::Derivative(DataType x) {
    return derivative_(x);
}

project::Vector project::ActivationFunction::Calc(Vector vector) {
    vector = vector.unaryExpr([this](DataType x) { return Calc(x); });
    return vector;
}

project::Matrix project::ActivationFunction::Derivative(Vector vector) {
    vector = vector.unaryExpr([this](DataType x) { return Derivative(x); });
    return vector.asDiagonal();
}

project::DataType project::act_func_options::Sigmoid::Calc(DataType x) {
    return 1.0 / (1.0 + exp(-x));
}
project::DataType project::act_func_options::Sigmoid::Derivative(DataType x) {
    return exp(x) / (exp(2.0 * x) + 2.0 * exp(x) + 1.0);
}

project::DataType project::act_func_options::Tanh::Calc(DataType x) {
    return tanh(x);
}
project::DataType project::act_func_options::Tanh::Derivative(DataType x) {
    return 1.0 / (cosh(x) * cosh(x));
}

project::DataType project::act_func_options::ReLU::Calc(DataType x) {
    return (x > 0) ? x : 0;
}
project::DataType project::act_func_options::ReLU::Derivative(DataType x) {
    return (x > 0) ? 1.0 : 0;
}

project::DataType project::act_func_options::Linear::Calc(DataType x) {
    return x;
}
project::DataType project::act_func_options::Linear::Derivative(DataType x) {
    return 1.0;
}
