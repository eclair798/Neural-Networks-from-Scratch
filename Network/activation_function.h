#pragma once

#include "definitions.h"

namespace project {

enum class AFName { Sigmoid, Tanh, ReLU, Linear, Softmax };

class ActivationFunction {
public:
    ActivationFunction() = default;
    ActivationFunction(FuncAct calc, FuncDerivativeAct der_calc);
    static ActivationFunction Make(AFName name);

    //    DataType Calc(DataType x) const;
    //    DataType Derivative(DataType x) const;
    Vector Calc(const Vector& vector) const;
    Matrix Derivative(const Vector& vector) const;
    Matrix CalcMatrix(const Matrix& matrix) const;

private:
    FuncAct calc_;
    FuncDerivativeAct derivative_;
};

namespace act_func_options {
class Sigmoid {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& vector);
    static Matrix Derivative(const Vector& vector);
};
class Tanh {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& vector);
    static Matrix Derivative(const Vector& vector);
};
class ReLU {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& vector);
    static Matrix Derivative(const Vector& vector);
};
class Linear {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& vector);
    static Matrix Derivative(const Vector& vector);
};

class Softmax {
public:
    static Vector Calc(const Vector& vector);
    static Matrix Derivative(const Vector& vector);
};

}  // namespace act_func_options
}  // namespace project
