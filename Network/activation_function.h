#pragma once

#include "../eigen/Eigen/Eigen"
#include "../EigenRand/EigenRand/EigenRand"

namespace project {

class ActivationFunction {
public:
    virtual double Calc(double x) {
    }

    virtual double Derivative(double x) {
    }

    Eigen::VectorXd Calc(Eigen::VectorXd vector);

    Eigen::MatrixXd Derivative(Eigen::VectorXd vector);
};

namespace act_func_options {

class Sigmoid : public ActivationFunction {
    double Calc(double x) override;
    double Derivative(double x) override;
};

class Tanh : public ActivationFunction {
    double Calc(double x) override;
    double Derivative(double x) override;
};

class ReLU : public ActivationFunction {
    double Calc(double x) override;
    double Derivative(double x) override;
};

class Linear : public ActivationFunction {
    double Calc(double x) override;
    double Derivative(double x) override;
};

}  // namespace act_func_options

}  // namespace project
