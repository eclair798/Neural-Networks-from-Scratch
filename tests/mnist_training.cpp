#pragma once

#include "mnist_training.h"

namespace project {

using ImagesContainer = std::vector<std::vector<uint8_t>>;
using LabelsContainer = std::vector<uint8_t>;

namespace {

DataType NormPixel(size_t pixel) {
    static constexpr const DataType kNorm = 255.0;
    return static_cast<double>(pixel) / kNorm;
}

//DataType NormPixel(size_t pixel) {
//    static constexpr const DataType kThrashHold = 75.0;
//    DataType result = pixel > kThrashHold ? 1.0 : 0.0;
//    return result;
//}

void PutInputVec(const ImagesContainer& images, Matrix& input, Index num_input_pixels, Index i) {
    for (Index j = 0; j < num_input_pixels; ++j) {
        input(j, i) = NormPixel(images[i][j]);
    }
}

void PutOutputVec(const LabelsContainer& labels, Matrix& output, Index i) {
    static constexpr const DataType kFlag = 1.0;
    output(static_cast<Index>(labels[i]), i) = kFlag;
}

}  // namespace

DataSet MnistTesting::GetMnistData(Index train_size) {
    auto mnist_dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");

    Index num_train_images = std::min(train_size, kMnistTrainDataSize);
    Index num_test_images = mnist_dataset.test_images.size();

    Index num_input_pixels = mnist_dataset.training_images[0].size();
    Index num_output_pixels = 10;

    Data train({num_input_pixels, num_train_images}, {num_output_pixels, num_train_images});
    Data test({num_input_pixels, num_test_images}, {num_output_pixels, num_test_images});

    const ImagesContainer& training_images = mnist_dataset.training_images;
    const LabelsContainer& training_labels = mnist_dataset.training_labels;
    const ImagesContainer& test_images = mnist_dataset.test_images;
    const LabelsContainer& test_labels = mnist_dataset.test_labels;

    for (Index i = 0; i < num_train_images; ++i) {
        PutInputVec(training_images, train.input_vectors, num_input_pixels, i);
        PutOutputVec(training_labels, train.output_vectors, i);
    }

    for (Index i = 0; i < num_test_images; ++i) {
        PutInputVec(test_images, test.input_vectors, num_input_pixels, i);
        PutOutputVec(test_labels, test.output_vectors, i);
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
    Path path1 = "../../tests/params.bin";
    Path path2 = "../../tests/params2.bin";
    Path path3 = "../../tests/params3.bin";
    Path path4 = "../../tests/params4.bin";  // 784 -- AFName::ReLU -- 32 -- AFName::Softmax -- 10

    constexpr Index kDataSize = 60000;
    DataSet dataset(GetMnistData(kDataSize));

    const Sizes k_layer_sizes = {dataset.num_input_pixels, 32, dataset.num_output_pixels};
    Path input_path = path1;
    Net net(k_layer_sizes, {AFName::ReLU, AFName::Softmax}, input_path);

    constexpr LFName kLFName = LFName::CrossEntropy;
    constexpr Index kIterCount = 0;
    constexpr DataType kIlr = 0.01;
    constexpr DataType kDecay = 0.01;
    Path output_path = "";
    MnistTesting::Train(net, dataset, kIterCount, kIlr, kDecay, kLFName, output_path);
    DataType accuracy = CalcAccuracy(net, dataset);
    std::cout << "Accuracy of Neural Network: " << accuracy << "\n\n";
}

}  // namespace project
