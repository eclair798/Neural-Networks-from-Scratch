#pragma once

#include "loss_function.h"
#include "layer.h"

namespace project {

class Net {
    struct Delta {
        Matrix delta_a;
        Vector delta_b;
    };

    using Layers = std::vector<Layer>;
    using AFNames = std::vector<AFName>;
    using Calculations = std::vector<Matrix>;
    using Deltas = std::vector<Delta>;

public:
    struct Info {
        DataType error_rate;
        Counter iterations_count;
    };

    Net(Sizes layer_sizes, const AFNames& act_funcs);
    Info Train(const Data& train_data, const Data& test_data, const LFName& dist_func,
               DataType eps = 0.01, Counter max_iter = 10, Index batches_count = 50,
               bool print_info = true);
    Vector Calc(const Vector& x) const;
    Matrix Calc(const Matrix& x) const;

private:
    Deltas GetCorrections(const Data& data) const;
    Layers layers_;
    LossFunction dist_func_;
};

}  // namespace project
