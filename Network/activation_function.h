#pragma once

#include "details.h"

namespace project {

class ActivationFunction {
public:
    ActivationFunction() = default;

    ActivationFunction(FuncAct calc, FuncAct der_calc) : calc_(calc), derivative_(der_calc) {
    }

    DataType Calc(DataType x);

    DataType Derivative(DataType x);

    Vector Calc(Eigen::VectorXd vector);

    Matrix Derivative(Eigen::VectorXd vector);

private:
    FuncAct calc_;
    FuncAct derivative_;
};

namespace act_func_options {

class Sigmoid {
public:
    static DataType Calc(DataType x);
    static DataType Derivative(DataType x);
};

class Tanh {
public:
    static DataType Calc(DataType x);
    static DataType Derivative(DataType x);
};

class ReLU {
public:
    static DataType Calc(DataType x);
    static DataType Derivative(DataType x);
};

class Linear {
public:
    static DataType Calc(DataType x);
    static DataType Derivative(DataType x);
};

}  // namespace act_func_options

}  // namespace project
