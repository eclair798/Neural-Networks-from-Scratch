#pragma once

#include "loss_function.h"
#include "layer.h"
#include "parameters_rw.h"

namespace project {

enum class PI { PrintInfo, DoNotPrintInfo };

class Net {

    using Layers = std::vector<Layer>;
    using AFNames = std::vector<AFName>;
    using Calculations = std::vector<Matrix>;
    using Deltas = std::vector<Parameter>;

public:
    struct Info {
        DataType error_rate;
        Index iterations_count;
    };

    Net(const Sizes& layer_sizes, const AFNames& act_funcs, const Path& input_path = "");
    Info Train(const Data& train_data, const LFName& dist_func, DataType error = kDefaultError,
               Index max_iter = kDefaultMaxIter, DataType initial_learning_rate = kDefaultInitLR, DataType decay = kDefaultDecay,
               Index batch_size = kDefaultBatchSize, PI print_info = kDefaultPI);
    void SaveParams(const Path& output_path = "");
    Vector Calc(const Vector& x) const;
    Matrix Calc(const Matrix& xs) const;

private:
    static constexpr const LFName kDefaultLFName = LFName::MSE;
    static constexpr const DataType kDefaultError = 0.01;
    static constexpr const Index kDefaultMaxIter = 10;
    static constexpr const DataType kDefaultInitLR = 0.1;
    static constexpr const DataType kDefaultDecay = 0;
    static constexpr const Index kDefaultBatchSize = 64;
    static constexpr const PI kDefaultPI = PI::PrintInfo;
    Deltas GetCorrections(const Data& data, LossFunction dist_func) const;
    Layers layers_;
    //    LossFunction dist_func_;
};

}  // namespace project
