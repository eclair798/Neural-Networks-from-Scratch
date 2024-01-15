#pragma one

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

namespace project {

class DistanceFunction {
public:
    virtual int Dist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) {
        return 0;
    }

    int Dist(Eigen::MatrixXd x, Eigen::MatrixXd y);

    virtual Eigen::RowVectorXd DerivativeDist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) {
        return Eigen::RowVectorXd();
    }

    Eigen::MatrixXd DerivativeDist(Eigen::MatrixXd x, Eigen::MatrixXd y);
};

namespace dist_func_options {

class SquaredEuclidean : public DistanceFunction {
    int Dist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) override;

    Eigen::RowVectorXd DerivativeDist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) override;
};

class Manhattan : public DistanceFunction {
    int Dist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) override;

    Eigen::RowVectorXd DerivativeDist(Eigen::RowVectorXd x, Eigen::RowVectorXd y) override;
};

}  // namespace dist_func_options

}  // namespace project