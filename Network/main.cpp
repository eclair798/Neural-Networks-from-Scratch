#include <iostream>

#include "application.h"

int main(/*int argc, char* argv[]*/) {
    try {
        return project::MnistTesting::Run();
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << std::endl;
        return -1;
    } catch (...) {
        return -2;
    }
}
