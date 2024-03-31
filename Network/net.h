#pragma once

#include "loss_function.h"
#include "layer.h"

namespace project {

class Net {
    using Layers = std::vector<Layer>;
    using AFNames = std::vector<AFName>;
    using Calculations = std::vector<Matrix>;
    struct Delta {
        Matrix delta_a;
        Vector delta_b;
    };
    using Deltas = std::vector<Delta>;

    struct Info {
        DataType error_rate;
        Counter iterations_count;
    };
public:
    Net(Sizes layer_sizes, const AFNames& act_funcs);
    Info Train(const Data& data, DataType eps, Counter max_iter,
               /*Alg,*/ const LFName& dist_func, Index batches_count = 50);  // TODO
    Vector Calc(const Vector& x) const;
    Matrix Calc(const Matrix& x) const;

private:
    Deltas GetCorrections(const Data& data) const;

    Layers layers_;
    LossFunction dist_func_;
};

}  // namespace project
