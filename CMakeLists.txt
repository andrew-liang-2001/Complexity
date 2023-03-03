cmake_minimum_required(VERSION 3.21)
project(complexity)

set(CMAKE_CXX_STANDARD 14)
set(PYBIND11_PYTHON_VERSION 3.6)

find_package(pybind11 REQUIRED)

add_executable(complexity oslo.cpp oslo.h)
