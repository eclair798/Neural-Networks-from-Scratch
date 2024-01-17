#pragma one

#include "loss_function.h"

namespace project {

DataType project::DistanceFunction::Dist(Vector x, Vector y) {
    return dist_(x, y);
}
Vector project::DistanceFunction::DerivativeDist(Vector x, Vector y) {
    return derivative_dist_(x, y);
}

DataType project::DistanceFunction::Dist(Matrix x, Matrix y) {
    DataType distance = 0;
    Matrix::Index size = x.rows();
    for (Matrix::Index row_i = 0; row_i < size; ++row_i) {
        Vector x_i = x.row(row_i);
        Vector y_i = y.row(row_i);
        distance += Dist(x_i, y_i);
    }
    distance /= size;
    return distance;
}

Matrix project::DistanceFunction::DerivativeDist(Matrix x, Matrix y) {
    Matrix matrix_u(x.rows(), x.cols());
    Matrix::Index size = x.rows();
    for (Matrix::Index row_i = 0; row_i < size; ++row_i) {
        Vector x_i = x.row(row_i);
        Vector y_i = y.row(row_i);
        matrix_u.row(row_i) = DerivativeDist(x_i, y_i);
    }
    return matrix_u;
}

DataType project::dist_func_options::SquaredEuclidean::Dist(Vector x, Vector y) {
    return sqrt((x - y).norm());
}
Vector project::dist_func_options::SquaredEuclidean::DerivativeDist(Vector x, Vector y) {
    Vector vector_u = 2 * (x - y);
    return vector_u;
}

DataType project::dist_func_options::Manhattan::Dist(Vector x, Vector y) {
    return (x - y).lpNorm<1>();
}
Vector project::dist_func_options::Manhattan::DerivativeDist(Vector x, Vector y) {
    Vector vector_u = (x - y).unaryExpr([](DataType x) { return x > 0 ? 1.0 : -1.0; });
    return vector_u;
}

}  // namespace project