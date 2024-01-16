#pragma one

#include "details.h"

namespace project {

class DistanceFunction {
public:
    DistanceFunction() = default;

    DistanceFunction(FuncDist calc, FuncDerivativeDist der_calc)
        : dist_(calc), derivative_dist_(der_calc) {
    }

    DataType Dist(RowVector x, RowVector y);

    RowVector DerivativeDist(RowVector x, RowVector y);

    DataType Dist(Matrix x, Matrix y);

    Matrix DerivativeDist(Matrix x, Matrix y);

private:
    FuncDist dist_;
    FuncDerivativeDist derivative_dist_;
};

namespace dist_func_options {

class SquaredEuclidean {
public:
    static DataType Dist(RowVector x, RowVector y);
    static RowVector DerivativeDist(RowVector x, RowVector y);
};

class Manhattan {
public:
    static DataType Dist(RowVector x, RowVector y);
    static RowVector DerivativeDist(RowVector x, RowVector y);
};

}  // namespace dist_func_options

}  // namespace project