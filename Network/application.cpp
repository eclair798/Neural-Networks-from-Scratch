#include "application.h"

namespace project {

int Application::Run() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");

    int num_train_images = 100;
    //    int num_train_images = dataset.training_images.size();
    int num_test_images = dataset.test_images.size();

    int num_input_pixels = dataset.training_images[0].size();
    int num_output_pixels = 10;

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

    Net net({num_input_pixels, 20, num_output_pixels}, {AFName::ReLU, AFName::Softmax}, "",
            "/home/xubuntu/Study/Neural-Networks-from-Scratch/res.bin");
    Net::Info info = net.Train(train, test, LFName::CrossEntropy, 0.001, 5, 0.01, 0.1, 64);
    std::cout << "FINAL RESULT:\n"
              << "iterations: " << info.iterations_count << "\terror rate: " << info.error_rate
              << "\n";
    return 0;
}
}  // namespace project
