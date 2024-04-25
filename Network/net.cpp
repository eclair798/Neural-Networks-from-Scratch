#pragma once

#include "net.h"

namespace project {

namespace {

void ShuffleData(Data& data) {
    Index total_size = data.input_vectors.cols();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<Index> dist(0, total_size - 1);
    for (Index i = 0; i < total_size; ++i) {
        Index rand_index = dist(gen);
        data.input_vectors.col(i).swap(data.input_vectors.col(rand_index));
        data.output_vectors.col(i).swap(data.output_vectors.col(rand_index));
    }
}

Batches DivideIntoBatches(Data data, Index batch_size) {
    assert(data.input_vectors.cols() == data.output_vectors.cols() &&
           "The number of input and output vectors differs");
    ShuffleData(data);
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

Net::Net(Sizes layer_sizes, const AFNames& act_funcs, Path input_path) {
    assert(!layer_sizes.empty() && "There cannot be zero layers");
    assert(layer_sizes.size() == act_funcs.size() + 1 &&
           "The number of layers does not correspond to the number of activation functions");

    if (!input_path.empty()) {
        reader_ = std::unique_ptr<ParameterReader>(new ParameterReader(input_path));
    }
    assert(reader_ == nullptr ||
           (layer_sizes.size() - 1 == reader_->GetParamsCount()) &&
               "The number of layers does not correspond to the number of parameters in file");

    for (Counter i = 0; i < layer_sizes.size() - 1; ++i) {
        layers_.emplace_back(layer_sizes[i], layer_sizes[i + 1],
                             ActivationFunction::Make(act_funcs[i]));
        if (reader_ != nullptr) {
            Parameter param = reader_->ReadParam();
            assert(param.matrix_a.cols() == layer_sizes[i] &&
                   param.matrix_a.rows() == layer_sizes[i + 1] &&
                   param.vector_b.rows() == layer_sizes[i + 1] &&
                   "The size of layers does not correspond to the size of parameters in file");
            assert(param.matrix_a.allFinite() && "Not finite data");
            assert(param.vector_b.allFinite() && "Not finite data");
            layers_.back().SetParam(std::move(param.matrix_a), std::move(param.vector_b));
        } else {
            layers_.back().SetRandParam();
        }
    }
}
Net::Info Net::Train(const Data& train_data, const LFName& dist_func, DataType eps,
                     Counter max_iter, DataType initial_learning_rate, DataType decay,
                     Index batch_size, bool print_info, Path output_path) {
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

    Matrix train_res = Calc(train_data.input_vectors);
    assert(train_res.allFinite() && "Not finite data");

    DataType train_error_rate = dist_func_.Dist(train_res, train_data.output_vectors);
    assert(res.allFinite() && "Not finite data");

    auto start = std::chrono::high_resolution_clock::now();
    Counter iterations_count = 0;
    for (Counter epoch = 1; epoch <= max_iter; ++epoch) {
        for (const Data& batch : batches) {
            cur_deltas = GetCorrections(batch);
            for (Counter j = 0; j < layers_.size(); ++j) {
                if (epoch == 1) {
                    average_deltas[j].matrix_a = (cur_deltas[j].matrix_a / batches.size());
                    average_deltas[j].vector_b = (cur_deltas[j].vector_b / batches.size());
                } else {
                    average_deltas[j].matrix_a += (cur_deltas[j].matrix_a / batches.size());
                    average_deltas[j].vector_b += (cur_deltas[j].vector_b / batches.size());
                }
            }
        }

        learning_rate = initial_learning_rate / (1 + epoch * decay);

        for (Counter j = 0; j < layers_.size(); ++j) {
            layers_[j].CorrectA(average_deltas[j].matrix_a, learning_rate);
            layers_[j].CorrectB(average_deltas[j].vector_b, learning_rate);
        }
        iterations_count = epoch;

        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start);

        if (print_info) {

            Matrix my_train_output = Calc(train_data.input_vectors);

            assert(my_train_output.allFinite() && "Not finite data");
            assert(my_test_output.allFinite() && "Not finite data");
            train_error_rate = dist_func_.Dist(my_train_output, train_data.output_vectors);

            std::cout << "iteration: " << iterations_count
                      << ";\t train error rate: " << train_error_rate
                      << ";\t time from start: " << duration.count() << "\n\n";
        }
        if (train_error_rate < eps) {
            break;
        }
    }

    if (!output_path.empty()) {
        writer_ =
            std::unique_ptr<ParameterWriter>(new ParameterWriter(output_path, layers_.size()));
    }

    if (writer_ != nullptr) {
        for (Layer& layer : layers_) {
            writer_->WriteParam(layer.GetA(), layer.GetB());
        }
    }

    Matrix my_train_output = Calc(train_data.input_vectors);

    assert(my_train_output.allFinite() && "Not finite data");
    assert(my_test_output.allFinite() && "Not finite data");
    train_error_rate = dist_func_.Dist(my_train_output, train_data.output_vectors);

    return {train_error_rate, iterations_count};
}

Matrix Net::Calc(const Matrix& x) const {
    assert(x.rows() == layers_.front().GetInputSize() &&
           "Incorrect dimension of the input vectors");
    Matrix cur_x = x;
    for (const Layer& layer : layers_) {
        assert(cur_x.allFinite() && "Not finite data");
        cur_x = layer.Calc(cur_x);
        assert(cur_x.allFinite() && "Not finite data");
    }
    return cur_x;
}

Net::Deltas Net::GetCorrections(const Data& data) const {
    assert(data.input_vectors.rows() == layers_.front().GetInputSize() &&
           data.output_vectors.rows() == layers_.back().GetOutputSize() &&
           "Mismatch with the size of the specified layers");
    assert(data.input_vectors.cols() == data.output_vectors.cols() &&
           "The number of input and output vectors differs");

    assert(data.input_vectors.allFinite() && "Not finite data");
    assert(data.output_vectors.allFinite() && "Not finite data");

    Counter calc_sizes = layers_.size() + 1;
    Calculations calcs(calc_sizes);
    calcs.front() = data.input_vectors;
    for (Counter i = 1; i < calc_sizes; ++i) {
        calcs[i] = layers_[i - 1].Calc(calcs[i - 1]);
    }

    assert(calcs[calc_sizes - 1].allFinite() && "Not finite data");
    Matrix ui = dist_func_.Grad(calcs[calc_sizes - 1], data.output_vectors);
    assert(ui.allFinite() && "Not finite data");

    Deltas deltas(layers_.size());
    for (Counter i = layers_.size() - 1; i >= 0; --i) {
        Matrix delta_a = layers_[i].GetACorrection(ui, calcs[i]);
        assert(delta_a.allFinite() && "Not finite data");
        Vector delta_b = layers_[i].GetBCorrection(ui, calcs[i]);
        assert(delta_b.allFinite() && "Not finite data");
        deltas[i] = {delta_a, delta_b};
        ui = layers_[i].PushU(ui, calcs[i]);
    }
    return deltas;
}

}  // namespace project
