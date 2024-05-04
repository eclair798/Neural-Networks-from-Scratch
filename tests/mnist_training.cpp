#pragma once

#include "mnist_training.h"

namespace project {

namespace {

Vector GetInputVec() {
}

Vector GetOutputVec() {
}

}  // namespace

DataSet MnistTesting::GetMnistData(Index train_size) {
    static constexpr const DataType kNorm = 255.0;
    static constexpr const DataType kFlag = 1.0;
    auto mnist_dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");

    Index num_train_images = std::min(train_size, kMnistTrainDataSize);
    Index num_test_images = mnist_dataset.test_images.size();

    Index num_input_pixels = mnist_dataset.training_images[0].size();
    Index num_output_pixels = 10;

    Data train({num_input_pixels, num_train_images}, {num_output_pixels, num_train_images});
    Data test({num_input_pixels, num_test_images}, {num_output_pixels, num_test_images});

    for (size_t i = 0; i < num_train_images; ++i) {
        for (size_t j = 0; j < num_input_pixels; ++j) {
            train.input_vectors(j, i) =
                static_cast<double>(mnist_dataset.training_images[i][j]) / kNorm;
        }
        train.output_vectors(static_cast<Index>(mnist_dataset.training_labels[i]), i) = kFlag;
    }
    for (size_t i = 0; i < num_test_images; ++i) {
        for (size_t j = 0; j < num_input_pixels; ++j) {
            test.input_vectors(j, i) = static_cast<double>(mnist_dataset.test_images[i][j]) / kNorm;
        }
        test.output_vectors(static_cast<Index>(mnist_dataset.test_labels[i]), i) = kFlag;
    }

    return {num_input_pixels, num_output_pixels, num_train_images, num_test_images, train, test};
}

int MnistTesting::Train(Net& net, DataSet& dataset, Index iter_count,
                        DataType initial_learning_rate, DataType decay, LFName name,
                        const Path& path) {
    Net::Info info = net.Train(dataset.train, name, kDefaultError, iter_count,
                               initial_learning_rate, decay, kDefaultBatchSize, kDefaultPI);
    net.SaveParams(path);
    std::cout << "RESULT:\n"
              << "iterations: " << info.iterations_count << "\terror rate: " << info.error_rate
              << "\n";
    return 0;
}

DataType MnistTesting::CalcAccuracy(const Net& net, const DataSet& dataset) {

    const Matrix& test_outputs = net.Calc(dataset.test.input_vectors);
    const Matrix& expected_outputs = dataset.test.output_vectors;
    DataType hits_count = 0;
    for (Index i = 0; i < dataset.num_test_images; ++i) {
        Index ans_index;
        test_outputs.col(i).maxCoeff(&ans_index);
        Index correct_index;
        expected_outputs.col(i).maxCoeff(&correct_index);
        if (ans_index == correct_index) {
            hits_count += 1.0;
        }
    }
    return hits_count / dataset.num_test_images;
}

void MnistTesting::Run() {
    Path path = "../../tests/mnist_ReLU_Softmax_32n_95per.bin";
    Path path2 = "../../tests/params";

    constexpr Index kDataSize = 60000;
    DataSet dataset(GetMnistData(kDataSize));

    const Sizes k_layer_sizes = {dataset.num_input_pixels, 32, dataset.num_output_pixels};
    Net net(k_layer_sizes, {AFName::ReLU, AFName::Softmax});

    constexpr Index kIterCount = 5;
    constexpr DataType kIlr = 0.1;
    constexpr DataType kDecay = 0.01;
    MnistTesting::Train(net, dataset, kIterCount, kIlr, kDecay, LFName::MSE, path2);
    DataType accuracy = CalcAccuracy(net, dataset);
    std::cout << "Accuracy of Neural Network: " << accuracy << "\n\n";
}

}  // namespace project
