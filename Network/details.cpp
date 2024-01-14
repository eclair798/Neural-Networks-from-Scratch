#include "details.h"

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