#pragma once

#include <initializer_list>
#include <string>
#include <vector>
#include <memory>

#include "details.h"

namespace project {

class Net {
public:
    Net(const Sizes& layer_sizes, const FuncNames act_funcs);

    void Train(const Data& data, double eps, size_t max_iter, /*Alg,*/ FuncName dist_func);  // TODO

private:
    // TODO

    Sizes layer_sizes_;
    ActFuncs act_funcs_ = {};
    std::shared_ptr<DistanceFunction> dist_func_ = nullptr;

    double error_;
    size_t max_iter_;

    std::shared_ptr<Eigen::MatrixXd> data_x_ = nullptr;
    std::shared_ptr<Eigen::MatrixXd> data_y_ = nullptr;
};

}  // namespace project
