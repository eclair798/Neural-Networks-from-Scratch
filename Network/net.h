#pragma once

#include "loss_function.h"
#include "layer.h"
#include "parameters_rw.h"

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

    Net(Sizes layer_sizes, const AFNames& act_funcs, Path input_path = "");
    Info Train(const Data& train_data, const Data& test_data, const LFName& dist_func,
               DataType eps = 0.01, Counter max_iter = 250, DataType initial_learning_rate = 0.1,
               DataType decay = 0, Index batch_size = 64, bool print_info = true,
               Path output_path = "");
    Vector Calc(const Vector& x) const;
    Matrix Calc(const Matrix& x) const;

private:
    Deltas GetCorrections(const Data& data) const;
    Layers layers_;
    LossFunction dist_func_;
    std::unique_ptr<ParameterReader> reader_ = nullptr;
    std::unique_ptr<ParameterWriter> writer_ = nullptr;
};

}  // namespace project
