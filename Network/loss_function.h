#pragma once

#include "definitions.h"

namespace project {

enum class LFName { SquaredEuclidean, Manhattan };

class LossFunction {
public:
    LossFunction() = default;
    LossFunction(FuncDist calc, FuncGrad der_calc);
    static LossFunction Make(LFName name);

    DataType Dist(const Vector& x, const Vector& y) const;
    Vector Grad(const Vector& x, const Vector& y) const;
    DataType Dist(const Matrix& x, const Matrix& y) const;
    Matrix Grad(const Matrix& x, const Matrix& y) const;

private:
    FuncDist dist_;
    FuncGrad grad_;
};

namespace loss_func_options {
class SquaredEuclidean {
public:
    static DataType Dist(const Vector& x, const Vector& y);
    static Vector Grad(const Vector& x, const Vector& y);
};
class Manhattan {
public:
    static DataType Dist(const Vector& x, const Vector& y);
    static Vector Grad(const Vector& x, const Vector& y);
};
}  // namespace loss_func_options
}  // namespace project
