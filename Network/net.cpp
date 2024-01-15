#include "net.h"

project::Net::Net(const project::Sizes& layer_sizes, const project::FuncNames act_funcs)
    : layer_sizes_(layer_sizes) {
    details::SetActFuncs(act_funcs, act_funcs_);
}

void project::Net::Train(const project::Data& data, double eps, size_t max_iter,
                         project::FuncName dist_func) {
    details::CreateMatrix(data.first, data_x_);
    details::CreateMatrix(data.second, data_y_);
    details::SetDistFunc(dist_func, dist_func_);
    error_ = eps;
    max_iter_ = max_iter;

    // TODO
}
