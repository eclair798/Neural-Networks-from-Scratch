#pragma once

#include "parameters_rw.h"

namespace project {

ParameterReader::ParameterReader(const Path& input_path) : input_file_(input_path, std::ios::binary) {
    assert(input_file_ && "Problem with input file");
    input_file_.read(reinterpret_cast<char*>(&params_count_), sizeof(params_count_));
}
Index ParameterReader::GetParamsCount() {
    return params_count_;
}
Parameter ParameterReader::ReadParam() {
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

    return param;
}

ParameterWriter::ParameterWriter(const Path& output_path, Index count)
    : output_file_(std::ofstream(output_path, std::ios::binary)) {
    assert(output_file_ && "Problem with output file");
    params_count_ = count;
    output_file_.write(reinterpret_cast<const char*>(&params_count_), sizeof(params_count_));
}
Index ParameterWriter::GetParamsCount() {
    return params_count_;
}
void ParameterWriter::WriteParam(const Matrix& matrix_a, const Vector& vector_b) {
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
}

}  // namespace project

// namespace project
