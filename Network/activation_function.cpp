#pragma once

#include "activation_function.h"

double project::ActivationFunction::Calc(double x) {
    return calc_(x);
}

double project::ActivationFunction::Derivative(double x) {
    return derivative_(x);
}

project::Vector project::ActivationFunction::Calc(Vector vector) {
    vector = vector.unaryExpr([this](double x) { return Calc(x); });
    return vector;
}

project::Matrix project::ActivationFunction::Derivative(Vector vector) {
    vector = vector.unaryExpr([this](double x) { return Derivative(x); });
    return vector.asDiagonal();
}

double project::act_func_options::Sigmoid::Calc(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double project::act_func_options::Sigmoid::Derivative(double x) {
    return exp(x) / (exp(2.0 * x) + 2.0 * exp(x) + 1.0);
}

double project::act_func_options::Tanh::Calc(double x) {
    return tanh(x);
}
double project::act_func_options::Tanh::Derivative(double x) {
    return 1.0 / (cosh(x) * cosh(x));
}

double project::act_func_options::ReLU::Calc(double x) {
    return (x > 0) ? x : 0;
}
double project::act_func_options::ReLU::Derivative(double x) {
    return (x > 0) ? 1.0 : 0;
}

double project::act_func_options::Linear::Calc(double x) {
    return x;
}
double project::act_func_options::Linear::Derivative(double x) {
    return 1.0;
}
