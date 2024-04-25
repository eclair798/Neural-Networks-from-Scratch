#include "definitions.h"

#include <fstream>

namespace project {

class ParameterReader {
public:
    ParameterReader(Path input_path);
    Counter GetParamsCount();
    Parameter ReadParam();
    ~ParameterReader();

private:
    std::ifstream input_file_;
    Counter params_count_;
    Counter processed_params_count_ = 0;
};

class ParameterWriter {
public:
    ParameterWriter(Path output_path, Counter count);
    Counter GetParamsCount();
    void WriteParam(const Matrix& matrix_a, const Vector& vector_b);
    ~ParameterWriter();

private:
    std::ofstream output_file_;
    Counter params_count_;
    Counter processed_params_count_ = 0;
};

}  // namespace project
