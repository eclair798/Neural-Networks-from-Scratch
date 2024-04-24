#pragma once

#include "../Network/net.h"

#include "../mnist/include/mnist/mnist_reader.hpp"

namespace project {

struct DataSet {
    Index num_input_pixels;
    Index num_output_pixels;
    Index num_train_images;
    Index num_test_images;
    Data train;
    Data test;
};

class MnistTesting {
public:
    static int Run(/*int argc, char* argv[]*/);

    static DataSet GetMnistData(Index train_size = 0);

    static int Train(Net& net, DataSet& dataset, Counter iter_count, DataType ilr, DataType decay,
                     Path path);

    static DataType CalcAccuracy(Net& net, DataSet& dataset);
};
}  // namespace project
