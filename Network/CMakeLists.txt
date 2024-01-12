include_directories("../eigen")
include_directories("../EigenRand")

file(GLOB SOURCES "*.cpp")

add_executable(Networks ${SOURCES})
