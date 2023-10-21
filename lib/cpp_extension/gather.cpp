#include <stdlib.h>
#include <aio.h>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <errno.h>
#include <cstring>
#include <inttypes.h>
#include <ATen/ATen.h>
#include <pthread.h>
#include <iostream>
#include <vector>
#define ALIGNMENT 4096


torch::Tensor gather_mmap(torch::Tensor features, torch::Tensor idx, int64_t feature_dim){

    // open file
    int64_t feature_size = feature_dim*sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();
    float* result_buffer = (float*)aligned_alloc(ALIGNMENT, feature_size*num_idx);

    auto features_data = features.data_ptr<float>();
    auto idx_data = idx.data_ptr<int64_t>();

    #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;

        i = idx_data[n];
        memcpy(result_buffer+feature_dim*n, features_data+i*feature_dim, feature_size);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

    return result;

}


torch::Tensor gather_direct(std::string feature_file, torch::Tensor idx, int64_t feature_dim){

    // open file
    int feature_fd = open(feature_file.c_str(), O_RDONLY);

    int64_t feature_size = feature_dim*sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();

    float* read_buffer = (float*)aligned_alloc(ALIGNMENT, ALIGNMENT*2*atoi(getenv("NUM_THREADS")));
    float* result_buffer = (float*)aligned_alloc(ALIGNMENT, feature_size*num_idx);

    auto idx_data = idx.data_ptr<int64_t>();

    #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t read_size;

        i = idx_data[n];

        offset = i * feature_size;
        aligned_offset = offset&(long)~(ALIGNMENT-1);
        residual = offset - aligned_offset;

        if (residual+feature_size > ALIGNMENT){
            read_size = ALIGNMENT * 2;
        }
        else {
            read_size = ALIGNMENT;
        }

        if (pread(feature_fd, read_buffer+(ALIGNMENT*2*omp_get_thread_num())/sizeof(float), read_size, aligned_offset) == -1){
            fprintf(stderr, "ERROR: %s\n", strerror(errno));
        }
        memcpy(result_buffer+feature_dim*n, read_buffer+(ALIGNMENT*2*omp_get_thread_num()+residual)/sizeof(float), feature_size);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

    free(read_buffer);
    close(feature_fd);

    return result;

}


std::tuple<torch::Tensor, int64_t, torch::Tensor, torch::Tensor> gather_zerocopy(std::string feature_file, torch::Tensor idx, int64_t feature_dim, torch::Tensor cache, torch::Tensor cache_table){

    // open file
    int feature_fd = open(feature_file.c_str(), O_RDONLY);

    int64_t feature_size = feature_dim*sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();

    float* read_buffer = (float*)aligned_alloc(ALIGNMENT, ALIGNMENT*2*atoi(getenv("NUM_THREADS")));
    float* result_buffer = (float*)aligned_alloc(ALIGNMENT, feature_size*num_idx);

    auto idx_data = idx.data_ptr<int64_t>();
    auto cache_data = cache.data_ptr<float>();
    auto cache_table_data = cache_table.data_ptr<int32_t>();

    auto hit_idx_vec = std::vector<int64_t>();
    auto hit_pos_vec = std::vector<int64_t>();

    auto non_hit_vec = std::vector<int64_t>();

    int64_t hit_num = 0;
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t cache_entry;

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            hit_num += 1;
            hit_idx_vec.push_back(cache_entry);
            hit_pos_vec.push_back(n);
        }
        else {
            non_hit_vec.push_back(n);
        }
    }

    auto non_hit_num = non_hit_vec.size();
    #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
    for (int64_t n = 0; n < non_hit_num; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t read_size;

        i = idx_data[non_hit_vec[n]];
        offset = i * feature_size;
        aligned_offset = offset&(long)~(ALIGNMENT-1);
        residual = offset - aligned_offset;

        if (residual+feature_size > ALIGNMENT){
            read_size = ALIGNMENT * 2;
        }
        else {
            read_size = ALIGNMENT;
        }

        if (pread(feature_fd, read_buffer+(ALIGNMENT*2*omp_get_thread_num())/sizeof(float), read_size, aligned_offset) == -1){
            fprintf(stderr, "ERROR: %s\n", strerror(errno));
        }
        memcpy(result_buffer+feature_dim*n, read_buffer+(ALIGNMENT*2*omp_get_thread_num()+residual)/sizeof(float), feature_size);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {non_hit_vec.size(), feature_dim}, options);

    auto hit_idx = torch::from_blob(hit_idx_vec.data(), {hit_idx_vec.size()}, idx.options()).clone();
    auto hit_pos = torch::from_blob(hit_pos_vec.data(), {hit_pos_vec.size()}, idx.options()).clone();

    free(read_buffer);
    close(feature_fd);

    // std::cout << (double)hit_num / (double)num_idx << std::endl;

    return std::make_tuple(result, hit_num, hit_idx, hit_pos);

}


std::tuple<torch::Tensor, int64_t> gather_normal(std::string feature_file, torch::Tensor idx, int64_t feature_dim, torch::Tensor cache, torch::Tensor cache_table){

    // open file
    int feature_fd = open(feature_file.c_str(), O_RDONLY);

    int64_t feature_size = feature_dim*sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();

    float* read_buffer = (float*)aligned_alloc(ALIGNMENT, ALIGNMENT*2*atoi(getenv("NUM_THREADS")));
    float* result_buffer = (float*)aligned_alloc(ALIGNMENT, feature_size*num_idx);

    auto idx_data = idx.data_ptr<int64_t>();
    auto cache_data = cache.data_ptr<float>();
    auto cache_table_data = cache_table.data_ptr<int32_t>();

    int64_t hit_num = 0;
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t cache_entry;

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            hit_num += 1;
        }
    }

    #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t cache_entry;
        int64_t read_size;

        i = idx_data[n];
        cache_entry = cache_table_data[i];
        if (cache_entry >= 0) {
            memcpy(result_buffer+feature_dim*n, cache_data+cache_entry*feature_dim, feature_size);
        }
        else {
            offset = i * feature_size;
            aligned_offset = offset&(long)~(ALIGNMENT-1);
            residual = offset - aligned_offset;

            if (residual+feature_size > ALIGNMENT){
                read_size = ALIGNMENT * 2;
            }
            else {
                read_size = ALIGNMENT;
            }

            if (pread(feature_fd, read_buffer+(ALIGNMENT*2*omp_get_thread_num())/sizeof(float), read_size, aligned_offset) == -1){
                fprintf(stderr, "ERROR: %s\n", strerror(errno));
            }
            memcpy(result_buffer+feature_dim*n, read_buffer+(ALIGNMENT*2*omp_get_thread_num()+residual)/sizeof(float), feature_size);
        }
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

    free(read_buffer);
    close(feature_fd);

    // std::cout << (double)hit_num / (double)num_idx << std::endl;

    return std::make_tuple(result, hit_num);

}


PYBIND11_MODULE(gather, m) {
    m.def("gather_normal", &gather_normal, "direct gather with cache", py::call_guard<py::gil_scoped_release>());
    m.def("gather_zerocopy", &gather_zerocopy, "gather in zero-copy way with cache", py::call_guard<py::gil_scoped_release>());
    m.def("gather_mmap", &gather_mmap, "gather in mmap way no cache", py::call_guard<py::gil_scoped_release>());
    m.def("gather_direct", &gather_direct, "direct gather with no cache", py::call_guard<py::gil_scoped_release>());
}




