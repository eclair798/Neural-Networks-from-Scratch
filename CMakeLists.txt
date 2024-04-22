cmake_minimum_required(VERSION 3.0)
project(Neural-Networks)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#enable_testing()
#
#find_package(GTest REQUIRED)
add_subdirectory(eigen)
add_subdirectory(EigenRand)
add_subdirectory(Network)
add_subdirectory(tests)

#file(GLOB TEST_SOURCES tests/*.cpp)
#
#add_executable(project_tests ${TEST_SOURCES})

#target_link_libraries(project_tests GTest::GTest GTest::Main)
#gtest_discover_tests(project_tests)