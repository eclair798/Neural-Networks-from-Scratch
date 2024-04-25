#include "mnist_training.h"

namespace project {

DataSet MnistTesting::GetMnistData(Index train_size) {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");

    Index num_train_images = train_size == 0 || train_size > dataset.training_images.size()
                                 ? dataset.training_images.size()
                                 : train_size;
    Index num_test_images = dataset.test_images.size();

    Index num_input_pixels = dataset.training_images[0].size();
    Index num_output_pixels = 10;

    Data train({num_input_pixels, num_train_images}, {num_output_pixels, num_train_images});
    Data test({num_input_pixels, num_test_images}, {num_output_pixels, num_test_images});

    for (size_t i = 0; i < num_train_images; ++i) {
        for (size_t j = 0; j < num_input_pixels; ++j) {
            train.input_vectors(j, i) = static_cast<double>(dataset.training_images[i][j]) / 255.0;
        }
        train.output_vectors(static_cast<Counter>(dataset.training_labels[i]), i) = 1.0;
    }
    for (size_t i = 0; i < num_test_images; ++i) {
        for (size_t j = 0; j < num_input_pixels; ++j) {
            test.input_vectors(j, i) = static_cast<double>(dataset.test_images[i][j]) / 255.0;
        }
        test.output_vectors(static_cast<Counter>(dataset.test_labels[i]), i) = 1.0;
    }

    return {num_input_pixels, num_output_pixels, num_train_images, num_test_images, train, test};
}

int MnistTesting::Train(Net& net, DataSet& dataset, Counter iter_count = 10, DataType ilr = 0.01,
                        DataType decay = 0, Path path = "") {

    Net::Info info = net.Train(dataset.train, LFName::CrossEntropy, 0.001,
                               iter_count, ilr, decay, 128, true, path);
    std::cout << "RESULT:\n"
              << "iterations: " << info.iterations_count << "\terror rate: " << info.error_rate
              << "\n";
    return 0;
}

DataType MnistTesting::CalcAccuracy(Net& net, DataSet& dataset) {

    auto test_outputs = net.Calc(dataset.test.input_vectors);
    auto& expected_outputs = dataset.test.output_vectors;
    DataType hits_count = 0;
    for (Counter i = 0; i < dataset.num_test_images; ++i) {
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

int MnistTesting::Run() {
    Path path = "../../tests/mnist_ReLU_Softmax_32n_95per.bin";
    Path path2 = "../../tests/params";

    DataSet dataset(GetMnistData(10000));

    Net net({dataset.num_input_pixels, 32, dataset.num_output_pixels},
            {AFName::ReLU, AFName::Softmax}, path);

        MnistTesting::Train(net, dataset, 10, 0.1, 0.01, path2);
    DataType accuracy = CalcAccuracy(net, dataset);
    std::cout << "Accuracy of Neural Network: " << accuracy << "\n\n";

    return 0;
}

}  // namespace project
