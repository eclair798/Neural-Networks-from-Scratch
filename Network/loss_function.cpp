#pragma one

#include "loss_function.h"

int project::DistanceFunction::Dist(Eigen::MatrixXd x, Eigen::MatrixXd y) {
    int dist = 0;
    Eigen::MatrixXd::Index size = x.rows();
    for (Eigen::MatrixXd::Index row_i = 0; row_i < size; ++row_i) {
        Eigen::RowVectorXd x_i = x.row(row_i);
        Eigen::RowVectorXd y_i = y.row(row_i);
        dist += Dist(x_i, y_i);
    }
    dist /= size;
    return dist;
}

Eigen::MatrixXd project::DistanceFunction::DerivativeDist(Eigen::MatrixXd x, Eigen::MatrixXd y) {
    Eigen::MatrixXd matrix_u(x.rows(), x.cols());
    Eigen::MatrixXd::Index size = x.rows();
    for (Eigen::MatrixXd::Index row_i = 0; row_i < size; ++row_i) {
        Eigen::RowVectorXd x_i = x.row(row_i);
        Eigen::RowVectorXd y_i = y.row(row_i);
        matrix_u.row(row_i) = DerivativeDist(x_i, y_i);
    }
}

int project::dist_func_options::SquaredEuclidean::Dist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) {
    return sqrt((x - y).norm());
}
Eigen::RowVectorXd project::dist_func_options::SquaredEuclidean::DerivativeDist(
    Eigen::RowVectorXd x, Eigen::RowVectorXd y) {
    Eigen::RowVectorXd vector_u = 2 * (x - y);
    return vector_u;
}

int project::dist_func_options::Manhattan::Dist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) {
    return (x - y).lpNorm<1>();
}
Eigen::RowVectorXd project::dist_func_options::Manhattan::DerivativeDist(Eigen::RowVectorXd x,
                                                                         Eigen::RowVectorXd y) {
    Eigen::RowVectorXd vector_u = (x - y).unaryExpr([](double x) { return x > 0 ? 1.0 : -1.0; });
    return vector_u;
}
