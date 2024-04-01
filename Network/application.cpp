#include "application.h"

namespace project {
int Application::Run() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../../mnist");

    int num_train_images = dataset.training_images.size();
    int num_test_images = dataset.test_images.size();

    int num_input_pixels = dataset.training_images[0].size();
    int num_output_pixels = 1;

    Data train({num_input_pixels, num_train_images}, {num_output_pixels, num_train_images});
    Data test({num_input_pixels, num_test_images}, {num_output_pixels, num_test_images});

    for (size_t i = 0; i < num_train_images; ++i) {
        for (size_t j = 0; j < num_input_pixels; ++j) {
            train.input_vectors(j, i) = static_cast<double>(dataset.training_images[i][j]) / 255.0;
        }
        train.output_vectors(0, i) = static_cast<double>(dataset.training_labels[i]);
    }
    for (size_t i = 0; i < num_test_images; ++i) {
        for (size_t j = 0; j < num_input_pixels; ++j) {
            test.input_vectors(j, i) = static_cast<double>(dataset.test_images[i][j]) / 255.0;
        }
        test.output_vectors(0, i) = static_cast<double>(dataset.test_labels[i]);
    }

    Net net({num_input_pixels, 5, 4, 3, num_output_pixels},
            {AFName::ReLU, AFName::ReLU, AFName::ReLU, AFName::ReLU});
    Net::Info info = net.Train(train, test, LFName::SquaredEuclidean, 0.1, 250);
    std::cout << "FINAL RESULT:\n"
              << "iterations: " << info.iterations_count << "\terror rate: " << info.error_rate
              << "\n";
    return 0;
}
}  // namespace project
