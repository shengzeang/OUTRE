#include <stdlib.h>
#include <torch/extension.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <iostream>

void tensor_free(torch::Tensor t){
    auto t_data = t.data_ptr();

    // std::cout << t_data << std::endl;

    free(t_data);

    return;
}

void tensor_ptr_free(void* ptr){
    free((void*)ptr);

    return;
}

void* get_tensor_ptr(torch::Tensor t){
    auto t_data = t.data_ptr();

    return (void*)t_data;
}

PYBIND11_MODULE(free, m) {
    m.def("tensor_free", &tensor_free);
    m.def("tensor_ptr_free", &tensor_ptr_free);
    m.def("get_tensor_ptr", &get_tensor_ptr);
}
