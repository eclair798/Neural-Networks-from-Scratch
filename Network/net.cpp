#pragma once

#include "net.h"

namespace project {

namespace {
Batches DivideIntoBatches(const Data& data, Index batch_size) {
    assert(data.input_vectors.cols() == data.output_vectors.cols() &&
           "The number of input and output vectors differs");
    Index total_size = data.input_vectors.cols();
    batch_size = std::min(total_size, batch_size);
    Index count = total_size / batch_size;
    if (total_size % batch_size > 0) {
        ++count;
    }
    Batches batches;
    for (Counter i = 0; i < count; ++i) {
        Index from = i * batch_size;
        Index to = from + batch_size - 1;
        to = std::min(to, total_size - 1);
        Index input_dim = data.input_vectors.rows();
        Index output_dim = data.output_vectors.rows();
        Matrix input = data.input_vectors.block(0, from, input_dim, to - from + 1);
        Matrix output = data.output_vectors.block(0, from, output_dim, to - from + 1);
        batches.emplace_back(input, output);
    }
    return batches;
}

}  // namespace

Net::Net(Sizes layer_sizes, const AFNames& act_funcs, Path input_path, Path output_path)
    : params_handler_(input_path, output_path) {
    assert(!layer_sizes.empty() && "There cannot be zero layers");
    assert(layer_sizes.size() == act_funcs.size() + 1 &&
           "The number of layers does not correspond to the number of activation functions");
    assert(!params_handler_.CanRead() ||
           (layer_sizes.size() - 1 == params_handler_.GetParamsCount()) &&
               "The number of layers does not correspond to the number of parameters in file");
    for (Counter i = 0; i < layer_sizes.size() - 1; ++i) {
        layers_.emplace_back(layer_sizes[i], layer_sizes[i + 1],
                             ActivationFunction::Make(act_funcs[i]));
        if (params_handler_.CanRead()) {
            Parameter param = params_handler_.ReadParam();
            layers_.back().SetParam(std::move(param.matrix_a), std::move(param.vector_b));
        } else {
            layers_.back().SetRandParam();
        }
    }
}
Net::Info Net::Train(const Data& train_data, const Data& test_data, const LFName& dist_func,
                     DataType eps, Counter max_iter, DataType initial_learning_rate, DataType decay,
                     Index batch_size, bool print_info) {
    assert(train_data.input_vectors.rows() == layers_.front().GetInputSize() &&
           train_data.output_vectors.rows() == layers_.back().GetOutputSize() &&
           "Mismatch with the size of the specified layers");
    assert(train_data.input_vectors.cols() == train_data.output_vectors.cols() &&
           "The number of input and output vectors differs");

    dist_func_ = LossFunction::Make(dist_func);
    Deltas average_deltas(layers_.size());
    Deltas cur_deltas;
    Batches batches = DivideIntoBatches(train_data, batch_size);
    DataType learning_rate;

    Matrix res = Calc(train_data.input_vectors);
    DataType error_rate = dist_func_.Dist(res, train_data.output_vectors);

    auto start = std::chrono::high_resolution_clock::now();
    Counter iterations_count = 0;
    for (Counter epoch = 1; epoch <= max_iter; ++epoch) {
        for (const Data& batch : batches) {
            cur_deltas = GetCorrections(batch);
            for (Counter j = 0; j < layers_.size(); ++j) {
                if (epoch == 1) {
                    average_deltas[j].delta_a = (cur_deltas[j].delta_a / batches.size());
                    average_deltas[j].delta_b = (cur_deltas[j].delta_b / batches.size());
                } else {
                    average_deltas[j].delta_a += (cur_deltas[j].delta_a / batches.size());
                    average_deltas[j].delta_b += (cur_deltas[j].delta_b / batches.size());
                }
            }
        }

        learning_rate = initial_learning_rate / (1 + epoch * decay);

        for (Counter j = 0; j < layers_.size(); ++j) {
            layers_[j].CorrectA(average_deltas[j].delta_a, learning_rate);
            layers_[j].CorrectB(average_deltas[j].delta_b, learning_rate);
        }
        iterations_count = epoch;

        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);

        if (print_info) {
            Matrix my_test_output = Calc(test_data.input_vectors);

            std::cout << my_test_output(7, 0) << "\n\n";

            error_rate = dist_func_.Dist(my_test_output, test_data.output_vectors);

            std::cout << "iteration: " << iterations_count << ";\t error rate: " << error_rate
                      << ";\t time from start: " << duration.count() << "\n\n";
        }
        if (error_rate < eps) {
            break;
        }
    }
    if (params_handler_.CanWrite()) {
        params_handler_.WriteHead(layers_.size());
        for (Layer& layer : layers_) {
            params_handler_.WriteParam(std::move(layer.GetA()), std::move(layer.GetB()));
        }
    }
    return {error_rate, iterations_count};
}

Vector Net::Calc(const Vector& x) const {
    assert(x.rows() == layers_.front().GetInputSize() && "Incorrect dimension of the input vector");
    Vector cur_x = x;
    for (const Layer& layer : layers_) {
        cur_x = layer.Calc(cur_x);
    }
    return cur_x;
}
Matrix Net::Calc(const Matrix& x) const {
    assert(x.rows() == layers_.front().GetInputSize() &&
           "Incorrect dimension of the input vectors");
    Matrix cur_x = x;
    for (const Layer& layer : layers_) {
        cur_x = layer.Calc(cur_x);
    }
    return cur_x;
}

Net::Deltas Net::GetCorrections(const Data& data) const {
    assert(data.input_vectors.rows() == layers_.front().GetInputSize() &&
           data.output_vectors.rows() == layers_.back().GetOutputSize() &&
           "Mismatch with the size of the specified layers");
    assert(data.input_vectors.cols() == data.output_vectors.cols() &&
           "The number of input and output vectors differs");

    Counter calc_sizes = layers_.size() + 1;
    Calculations calcs(calc_sizes);
    calcs.front() = data.input_vectors;
    for (Counter i = 1; i < calc_sizes; ++i) {
        calcs[i] = layers_[i - 1].Calc(calcs[i - 1]);
    }
    Matrix ui = dist_func_.Grad(calcs[calc_sizes - 1], data.output_vectors);
    Deltas deltas(layers_.size());
    for (Counter i = layers_.size() - 1; i >= 0; --i) {
        Matrix delta_a = layers_[i].GetACorrection(ui, calcs[i]);
        Vector delta_b = layers_[i].GetBCorrection(ui, calcs[i]);
        deltas[i] = {delta_a, delta_b};
        ui = layers_[i].PushU(ui, calcs[i]);
    }
    return deltas;
}

}  // namespace project
