#include "parameters_handler.h"

namespace project {

ParametersHandler::ParametersHandler(Path input_path, Path output_path)
    : output_path_(output_path) {
    can_read_ = !input_path.empty();
    can_write_ = !output_path.empty();
    if (can_read_) {
        input_file_.open(input_path, std::ios::binary);
        assert(input_file_ && "Problem with input file");
        input_file_.read(reinterpret_cast<char*>(&params_count_), sizeof(params_count_));
        been_read_ = (params_count_ == 0);
    }
    processed_params_count_ = 0;
}

bool ParametersHandler::CanRead() {
    return can_read_;
}
bool ParametersHandler::CanWrite() {
    return can_write_;
}
Counter ParametersHandler::GetParamsCount() {
    return params_count_;
}

Parameter ParametersHandler::ReadParam() {
    assert(processed_params_count_ != params_count_ && "Trying to read empty file");
    Parameter param;

    Index rows, cols;
    input_file_.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input_file_.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    param.matrix_a.resize(rows, cols);
    input_file_.read(reinterpret_cast<char*>(param.matrix_a.data()),
                     rows * cols * sizeof(DataType));

    Index size;
    input_file_.read(reinterpret_cast<char*>(&size), sizeof(size));

    param.vector_b.resize(size);
    input_file_.read(reinterpret_cast<char*>(param.vector_b.data()), size * sizeof(DataType));

    ++processed_params_count_;
    if (processed_params_count_ == params_count_) {
        been_read_ = true;
        input_file_.close();
    }
    return param;
}

void ParametersHandler::WriteHead(Counter count) {
    assert(can_write_ && "Can not write parameters into files");
    params_count_ = count;
    processed_params_count_ = 0;
    output_file_.open(output_path_, std::ios::binary);
    assert(output_file_ && "Problem with output file");
    output_file_.write(reinterpret_cast<char*>(&count), sizeof(count));
}
void ParametersHandler::WriteParam(Matrix matrix_a, Vector vector_b) {
    assert(can_write_ && "Can not write parameters into files");
    assert(processed_params_count_ != params_count_ && "Trying to write into full file");
    Index rows = matrix_a.rows();
    Index cols = matrix_a.cols();
    output_file_.write(reinterpret_cast<char*>(&rows), sizeof(rows));
    output_file_.write(reinterpret_cast<char*>(&cols), sizeof(cols));
    output_file_.write(reinterpret_cast<const char*>(matrix_a.data()),
                       rows * cols * sizeof(DataType));

    Index size = vector_b.rows();
    output_file_.write(reinterpret_cast<char*>(&size), sizeof(size));
    output_file_.write(reinterpret_cast<const char*>(vector_b.data()), size * sizeof(DataType));

    ++processed_params_count_;
}
ParametersHandler::~ParametersHandler() {
    if (input_file_.is_open()) {
        input_file_.close();
    }
    if (output_file_.is_open()) {
        output_file_.close();
    }
}

}  // namespace project

// namespace project
