#pragma once

#include "definitions.h"

namespace nn {

enum class LFName { MSE, Manhattan, CrossEntropy };

class LossFunction {
public:
    using FuncDist = std::function<DataType(const Vector&, const Vector&)>;
    using FuncGrad = std::function<RowVector(const Vector&, const Vector&)>;

    LossFunction(FuncDist calc, FuncGrad der_calc);
    static LossFunction Make(LFName name);

    DataType Dist(const Vector& x, const Vector& y) const;
    Vector Grad(const Vector& x, const Vector& y) const;
    DataType Dist(const Matrix& xs, const Matrix& ys) const;
    Matrix Grad(const Matrix& xs, const Matrix& ys) const;

private:
    FuncDist dist_;
    FuncGrad grad_;
};

namespace loss_func_options {
class MSE {
public:
    static DataType Dist(const Vector& x, const Vector& y);
    static Vector Grad(const Vector& x, const Vector& y);
};

class Manhattan {
public:
    static DataType Dist(const Vector& x, const Vector& y);
    static Vector Grad(const Vector& x, const Vector& y);
};

class CrossEntropy {
public:
    static DataType Dist(const Vector& x, const Vector& y);
    static Vector Grad(const Vector& x, const Vector& y);
};
}  // namespace loss_func_options
}  // namespace nn
