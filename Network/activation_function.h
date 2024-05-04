#pragma once

#include "definitions.h"

namespace project {

enum class AFName { Sigmoid, Tanh, ReLU, Linear, Softmax };

class ActivationFunction {
public:
    using Func = std::function<Vector(Vector)>;
    using FuncDerivative = std::function<Matrix(Vector)>;
    using FuncDerivativeDim1 = std::function<DataType(DataType)>;

    ActivationFunction() = default;
    ActivationFunction(
        Func calc, FuncDerivative der_calc, bool diag_der = true,
        FuncDerivativeDim1 der_dim1_calc = [](DataType x) { return 0; });
    static ActivationFunction Make(AFName name);
    Vector Calc(const Vector& x) const;
    Matrix Derivative(const Vector& x) const;
    Matrix CalcBatch(const Matrix& xs) const;
    Matrix DerivativeBatch(const Matrix& xs) const;
    bool IsDiagonalDerivative() const;

private:
    Func calc_;
    FuncDerivative derivative_;
    FuncDerivativeDim1 derivative_dim1_;
    bool is_diagonal_derivative_;
};

namespace act_func_options {
class Sigmoid {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& x);
    static Matrix Derivative(const Vector& x);
};
class Tanh {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& x);
    static Matrix Derivative(const Vector& x);
};
class ReLU {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& x);
    static Matrix Derivative(const Vector& x);
};
class Linear {
public:
    static DataType CalcDim1(DataType x);
    static DataType DerivativeDim1(DataType x);
    static Vector Calc(const Vector& x);
    static Matrix Derivative(const Vector& x);
};

class Softmax {
public:
    static Vector Calc(const Vector& x);
    static Matrix Derivative(const Vector& x);
};

}  // namespace act_func_options
}  // namespace project
