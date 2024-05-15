#pragma once

#include "loss_function.h"
#include "layer.h"
#include "parameters_rw.h"

namespace nn {

enum class PrintInfo { PrintInfo, DoNotPrintInfo };

class Net {
    using Layers = std::vector<Layer>;
    using Calculations = std::vector<Matrix>;
    using Deltas = std::vector<Parameter>;

public:
    using AFNames = std::vector<AFName>;

    struct Info {
        DataType error_rate;
        Index iterations_count;
    };

    Net(const Sizes& layer_sizes, const AFNames& act_funcs, const Path& input_path = "");
    Info Train(const Data& train_data, const LFName& dist_func = kDefaultLFName,
               DataType error = kDefaultError, Index max_iter = kDefaultMaxIter,
               DataType initial_learning_rate = kDefaultInitLR, DataType decay = kDefaultDecay,
               Index batch_size = kDefaultBatchSize, PrintInfo print_info = kDefaultPI);
    void SaveParams(const Path& output_path = "");
    Matrix Calc(const Matrix& xs) const;

private:
    static constexpr const LFName kDefaultLFName = LFName::MSE;
    static constexpr const DataType kDefaultError = 0.01;
    static constexpr const Index kDefaultMaxIter = 10;
    static constexpr const DataType kDefaultInitLR = 0.1;
    static constexpr const DataType kDefaultDecay = 0;
    static constexpr const Index kDefaultBatchSize = 64;
    static constexpr const PrintInfo kDefaultPI = PrintInfo::PrintInfo;
    Deltas GetCorrections(const Data& data, LossFunction dist_func) const;
    Layers layers_;
};

}  // namespace nn
