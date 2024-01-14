#pragma once

#include "net.h"

project::Net::Net(const project::Sizes& layer_sizes, const project::FuncNames act_funcs)
    : layer_sizes_(layer_sizes) {
    details::SetActFuncs(act_funcs, &act_funcs_);
}

void project::Net::Train(const project::Data& data, double eps, size_t max_iter,
                         project::FuncName dist_func) {
    details::CreateMatrix(data.first, data_x_);
    details::CreateMatrix(data.second, data_y_);
    details::SetDistFunc(dist_func, &dist_func_);
    error_ = eps;
    max_iter_ = max_iter;
}

void project::details::CreateMatrix(const project::DataVec& data,
                                    std::unique_ptr<Eigen::MatrixXd>& place) {
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

void project::details::SetActFuncs(FuncNames names, ActFuncs* place) {
    for (const FuncName& name : names) {
        place->push_back(act_functions[name]);
    }
}

void project::details::SetDistFunc(FuncName name, DistanceFunction* place) {
    *place = dist_functions[name];
}
