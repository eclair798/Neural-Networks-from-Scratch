#pragma once

#include "details.h"

namespace project {

class ActivationFunction {
public:
    ActivationFunction() = default;

    ActivationFunction(FuncAct calc, FuncAct der_calc) : calc_(calc), derivative_(der_calc) {
    }

    double Calc(double x);

    double Derivative(double x);

    Vector Calc(Eigen::VectorXd vector);

    Matrix Derivative(Eigen::VectorXd vector);

private:
    FuncAct calc_;
    FuncAct derivative_;
};

namespace act_func_options {

class Sigmoid {
public:
    static double Calc(double x);
    static double Derivative(double x);
};

class Tanh {
public:
    static double Calc(double x);
    static double Derivative(double x);
};

class ReLU {
public:
    static double Calc(double x);
    static double Derivative(double x);
};

class Linear {
public:
    static double Calc(double x);
    static double Derivative(double x);
};

}  // namespace act_func_options

}  // namespace project
