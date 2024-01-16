#pragma one

#include "loss_function.h"

int project::DistanceFunction::Dist(RowVector x, RowVector y) {
    return dist_(x, y);
}
project::RowVector project::DistanceFunction::DerivativeDist(RowVector x, RowVector y) {
    return derivative_dist_(x, y);
}

int project::DistanceFunction::Dist(Matrix x, Matrix y) {
    int distance = 0;
    Matrix::Index size = x.rows();
    for (Matrix::Index row_i = 0; row_i < size; ++row_i) {
        RowVector x_i = x.row(row_i);
        RowVector y_i = y.row(row_i);
        distance += Dist(x_i, y_i);
    }
    distance /= size;
    return distance;
}

project::Matrix project::DistanceFunction::DerivativeDist(Matrix x, Matrix y) {
    Matrix matrix_u(x.rows(), x.cols());
    Matrix::Index size = x.rows();
    for (Matrix::Index row_i = 0; row_i < size; ++row_i) {
        RowVector x_i = x.row(row_i);
        RowVector y_i = y.row(row_i);
        matrix_u.row(row_i) = DerivativeDist(x_i, y_i);
    }
    return matrix_u;
}

int project::dist_func_options::SquaredEuclidean::Dist(RowVector x, RowVector y) {
    return sqrt((x - y).norm());
}
project::RowVector project::dist_func_options::SquaredEuclidean::DerivativeDist(RowVector x,
                                                                                RowVector y) {
    RowVector vector_u = 2 * (x - y);
    return vector_u;
}

int project::dist_func_options::Manhattan::Dist(RowVector x, RowVector y) {
    return (x - y).lpNorm<1>();
}
project::RowVector project::dist_func_options::Manhattan::DerivativeDist(RowVector x, RowVector y) {
    RowVector vector_u = (x - y).unaryExpr([](double x) { return x > 0 ? 1.0 : -1.0; });
    return vector_u;
}
