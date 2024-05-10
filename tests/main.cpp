#include "mnist_training.h"
#include "except.h"

int main() {
    try {
        project::MnistTesting::Run();
    } catch (...) {
        except::React();
    }
    return 0;
}
