#pragma once

#include "definitions.h"

namespace project {

enum class AFName { Sigmoid, Tanh, ReLU, Linear };

class ActivationFunction {
public:
    ActivationFunction() = default;
    ActivationFunction(FuncAct calc, FuncAct der_calc);
    static ActivationFunction Make(AFName name);

    DataType Calc(DataType x) const;
    DataType Derivative(DataType x) const;
    Vector Calc(const Vector& vector) const;
    Matrix Derivative(const Vector& vector) const;

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
