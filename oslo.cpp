//
// Created by Andrew Liang on 03/03/2023.
//

#include "oslo.h"
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(MyLib, m) {
    m.doc() = "OsloModel written in C++ and wrapped in Python";
    py::class_<OsloModel>(m, "OsloModel
    .def(py::init<int>(), py::arg("L")
    .def("run", &OsloModel::run, py::call_guard<py::gil_scoped_release>())
    ;
}

int main() {
    int L = 1024;
    int runs = 10000;

    OsloModel model(L);
    for (int i = 0; i < runs; i++) {
        int s = model.run();
    }
    return 0;
}
