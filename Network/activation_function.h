#pragma once

#include "definitions.h"

namespace project {

enum class AFName { Sigmoid, Tanh, ReLU, Linear, Softmax };

class ActivationFunction {
public:
    ActivationFunction() = default;
    ActivationFunction(
        FuncAct calc, FuncDerivativeAct der_calc, bool diag_der = true,
        FuncDerivativeDim1Act der_dim1_calc = [](DataType x) { return 0; });
    static ActivationFunction Make(AFName name);
    Vector Calc(const Vector& vector) const;
    Matrix Derivative(const Vector& vector) const;
    Matrix CalcBatch(const Matrix& matrix) const;
    Matrix DerivativeBatch(const Matrix& matrix) const;
    bool IsDiagonalDerivative() const;

private:
    FuncAct calc_;
    FuncDerivativeAct derivative_;
    bool is_diagonal_derivative_;
    FuncDerivativeDim1Act derivative_dim1_;
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
