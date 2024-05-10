#pragma once

#include "definitions.h"

namespace project {

enum class AFName { Sigmoid, Tanh, ReLU, Linear, Softmax };

class ActivationFunction {
public:
    using Func = std::function<DataType(DataType)>;

    ActivationFunction() = default;
    ActivationFunction(Func calc, Func der_calc);

    DataType Calc(DataType xi) const;
    DataType Derivative(DataType xi) const;
    Vector Calc(const Vector& x) const;
    Matrix Derivative(const Vector& x) const;
    Matrix Calc(const Matrix& xs) const;
    Matrix Derivative(const Matrix& xs) const;

private:
    Func calc_;
    Func derivative_;
};

namespace act_func_options {
class Sigmoid {
public:
    static DataType Calc(DataType xi);
    static DataType Derivative(DataType xi);
};

class Tanh {
public:
    static DataType Calc(DataType xi);
    static DataType Derivative(DataType xi);
};

class ReLU {
public:
    static DataType Calc(DataType xi);
    static DataType Derivative(DataType xi);
};

class Linear {
public:
    static DataType Calc(DataType xi);
    static DataType Derivative(DataType xi);
};

class Softmax {
public:
    Softmax() = default;
    static Vector Calc(const Vector& x);
    static Matrix Derivative(const Vector& x);
    static Matrix Calc(const Matrix& xs);
};

}  // namespace act_func_options

using Sigma = std::variant<ActivationFunction, act_func_options::Softmax>;

Sigma AFMake(AFName name);

}  // namespace project