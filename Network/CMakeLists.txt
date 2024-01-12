file(GLOB SOURCES "*.cpp")

include_directories("../eigen")
include_directories("../EigenRand")

add_executable(Networks ${SOURCES})
