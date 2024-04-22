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

int MnistTesting::Training(Path path) {

    DataSet dataset(GetMnistData());

    Net net({dataset.num_input_pixels, 64, dataset.num_output_pixels},
            {AFName::ReLU, AFName::Softmax}, path);

    Net::Info info = net.Train(dataset.train, dataset.test, LFName::CrossEntropy, 0.001, 0, 0.01,
                               0.01, 128, true);
    std::cout << "RESULT:\n"
              << "iterations: " << info.iterations_count << "\terror rate: " << info.error_rate
              << "\n";
    return 0;
}

DataType MnistTesting::CalcAccuracy(Path path) {
    DataSet dataset(GetMnistData());

    Net net({dataset.num_input_pixels, 64, dataset.num_output_pixels},
            {AFName::ReLU, AFName::Softmax}, path);

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
        //        std::cout << "cur hits count: " << hits_count << " from " <<
        //        dataset.num_test_images << "\n";
    }
    return hits_count / dataset.num_test_images;
}

int MnistTesting::Run() {
    Path path = "../../params.bin";
    //    Path path = "";
    //    MnistTesting::Training(path);
    DataType accuracy = CalcAccuracy(path);
    std::cout << "Accuracy of Neural Network: " << accuracy << "\n\n";

    return 0;
}

}  // namespace project
