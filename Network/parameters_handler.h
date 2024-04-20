#include "definitions.h"

#include <fstream>

namespace project {

struct Parameter {
    Matrix matrix_a;
    Vector vector_b;
};

class ParametersHandler {
public:
    ParametersHandler(Path input_path, Path output_path);

    bool CanRead();
    bool CanWrite();
    Counter GetParamsCount();

    Parameter ReadParam();

    void WriteHead(Counter count);
    void WriteParam(Matrix matrix_a, Vector vector_b);

    ~ParametersHandler();

private:
    std::ifstream input_file_;
    std::ofstream output_file_;

    Path output_path_;

    Counter params_count_;
    Counter processed_params_count_ = 0;

    bool can_read_;
    bool can_write_;
    bool been_read_ = true;
};

}  // namespace project
