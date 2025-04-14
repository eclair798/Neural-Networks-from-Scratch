#pragma once

#include "except.h"

namespace except {
void React() {
    try {
        throw;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    } catch (...) {}
}
}  // namespace except
