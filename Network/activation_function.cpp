#pragma once

#include "activation_function.h"

Eigen::VectorXd project::ActivationFunction::Calc(Eigen::VectorXd vector) {
    vector = vector.unaryExpr([this](double x) { return Calc(x); });
    return vector;
}

Eigen::MatrixXd project::ActivationFunction::Derivative(Eigen::VectorXd vector) {
    vector = vector.unaryExpr([this](double x) { return Derivative(x); });
    return vector.asDiagonal();
}

double project::func_options::Sigmoid::Calc(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double project::func_options::Sigmoid::Derivative(double x) {
    return exp(x) / (exp(2.0 * x) + 2.0 * exp(x) + 1.0);
}

double project::func_options::Tanh::Calc(double x) {
    return tanh(x);
}
double project::func_options::Tanh::Derivative(double x) {
    return 1.0 / (cosh(x) * cosh(x));
}

double project::func_options::ReLU::Calc(double x) {
    return (x > 0) ? x : 0;
}
double project::func_options::ReLU::Derivative(double x) {
    return (x > 0) ? 1.0 : 0;
}

double project::func_options::Linear::Calc(double x) {
    return x;
}
double project::func_options::Linear::Derivative(double x) {
    return 1.0;
}
