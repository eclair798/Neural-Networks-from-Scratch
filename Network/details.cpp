#include "details.h"

namespace project::details {
std::map<FuncName, std::shared_ptr<ActivationFunction>> act_functions = {
    {"Sigmoid", std::shared_ptr<ActivationFunction>(new act_func_options::Sigmoid())},
    {"Tanh", std::shared_ptr<ActivationFunction>(new act_func_options::Tanh())},
    {"ReLU", std::shared_ptr<ActivationFunction>(new act_func_options::ReLU())},
    {"Linear", std::shared_ptr<ActivationFunction>(new act_func_options::Linear())}};

std::map<FuncName, std::shared_ptr<DistanceFunction>> dist_functions = {
    {"SquaredEuclidean",
     std::shared_ptr<DistanceFunction>(new dist_func_options::SquaredEuclidean())},
    {"Manhattan", std::shared_ptr<DistanceFunction>(new dist_func_options::Manhattan())}};
}  // namespace project::details

void project::details::CreateMatrix(const project::DataVec& data,
                                    std::shared_ptr<Eigen::MatrixXd>& place) {
    Eigen::MatrixXd::Index rows = data.size();
    Eigen::MatrixXd::Index cols = data.begin()->size();

    place = std::make_unique<Eigen::MatrixXd>(rows, cols);

    int i = 0;
    for (const auto& row : data) {
        int j = 0;
        for (const auto& element : row) {
            place->operator()(i, j) = element;
            ++j;
        }
        ++i;
    }
}

void project::details::SetActFuncs(FuncNames names, ActFuncs& place) {
    for (const FuncName& name : names) {
        place.push_back(act_functions[name]);
    }
}

void project::details::SetDistFunc(FuncName name, std::shared_ptr<DistanceFunction>& place) {
    place = dist_functions[name];
}