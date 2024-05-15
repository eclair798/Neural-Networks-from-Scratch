#include "mnist_training.h"
#include "except.h"

int main() {
    try {
        nn::MnistTesting::Run();
    } catch (...) {
        except::React();
    }
    return 0;
}
