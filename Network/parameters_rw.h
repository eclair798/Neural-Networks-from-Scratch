#pragma once

#include "definitions.h"

#include <fstream>

namespace project {

using Path = std::string;

class ParameterReader {
public:
    ParameterReader(const Path& input_path);
    Index GetParamsCount();
    Parameter ReadParam();
private:
    std::ifstream input_file_;
    Index params_count_;
};

class ParameterWriter {
public:
    ParameterWriter(const Path& output_path, Index count);
    Index GetParamsCount();
    void WriteParam(const Matrix& matrix_a, const Vector& vector_b);
private:
    std::ofstream output_file_;
    Index params_count_;
};

}  // namespace project
