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
#include <math.h>
#include <inttypes.h>
#include <ATen/ATen.h>
#include <pthread.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <random>
#include <mutex>
#include <chrono>

#include "bloom_filter.hpp"

#define ALIGNMENT 4096


int64_t load_neighbors_into_buffer(int col_fd, int64_t row_start, int64_t row_count, int64_t* buffer)
{
    int64_t size = (row_count*sizeof(int64_t) + 2*ALIGNMENT)&(long)~(ALIGNMENT-1);
    int64_t offset = row_start*sizeof(int64_t);
    int64_t aligned_offset = offset&(long)~(ALIGNMENT-1);

    if (pread(col_fd, buffer, size, aligned_offset) == -1)
        fprintf(stderr, "ERROR: %s\n", strerror(errno));

    return (offset-aligned_offset)/sizeof(int64_t);
}

std::tuple<int64_t*, int64_t*, int64_t> get_new_neighbor_buffer(int64_t row_count)
{
    int64_t size = (row_count*sizeof(int64_t) + 3*ALIGNMENT)&(long)~(ALIGNMENT-1);
    int64_t* neighbor_buffer = (int64_t*)malloc(size + ALIGNMENT);
    int64_t* aligned_neighbor_buffer = (int64_t*)(((long)neighbor_buffer+(long)ALIGNMENT)&(long)~(ALIGNMENT-1));

    return std::make_tuple(neighbor_buffer, aligned_neighbor_buffer, size/sizeof(int64_t));
}

std::tuple<int64_t*, int64_t*, int64_t> get_neighbors(int col_fd, int64_t st_pos, int64_t neighbor_count)
{
    // prepare buffer
    int64_t neighbor_buffer_size = 1<<15;
    int64_t* neighbor_buffer = (int64_t*)malloc(neighbor_buffer_size*sizeof(int64_t) + 2*ALIGNMENT);
    int64_t* aligned_neighbor_buffer = (int64_t*)(((long)neighbor_buffer+(long)ALIGNMENT)&(long)~(ALIGNMENT-1));

    if (neighbor_count > neighbor_buffer_size)
    {
        free(neighbor_buffer);
        std::tie(neighbor_buffer, aligned_neighbor_buffer, neighbor_buffer_size) = get_new_neighbor_buffer(neighbor_count);
    }
    int64_t start_offset = -1;
    if (neighbor_count > 0)
        start_offset = load_neighbors_into_buffer(col_fd,  st_pos, neighbor_count, aligned_neighbor_buffer);

    return std::make_tuple(neighbor_buffer, aligned_neighbor_buffer, start_offset);
}


void fennel_partition(torch::Tensor result, torch::Tensor cross_edge_num,
                        torch::Tensor csr_indptr, torch::Tensor csr_indices,
                        int64_t num_partitions, int64_t num_rounds,  int64_t num_train_nodes,
                        torch::Tensor train_mask, int patience, double imbalance_ratio, double gamma, torch::Tensor mapping)
{
    // fennel partition
    auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
    auto csr_indices_data = csr_indices.data_ptr<int64_t>();

    auto train_mask_data = train_mask.data_ptr<bool>();

    auto cross_edge_num_data = cross_edge_num.data_ptr<int64_t>();

    int64_t num_nodes = csr_indptr.numel() - 1;
    int64_t num_edges = csr_indices.numel();
    double alpha = pow((double)num_partitions, gamma-1) * (double)num_edges / pow((double)num_nodes, gamma);
    double max_partition_size = (double)num_nodes / (double)num_partitions * imbalance_ratio;

    double max_train_partition_size = (double)num_train_nodes / (double)num_partitions * imbalance_ratio;

    std::vector<std::vector<int64_t> > partition_has_nodes;
    for (int64_t i = 0; i < num_partitions; ++i)
        partition_has_nodes.push_back(std::vector<int64_t>());

    auto partition_assignment = std::vector<int64_t>(num_nodes, 0);
    auto partition_size = std::vector<int64_t>(num_partitions, 0);
    auto train_partition_size = std::vector<int64_t>(num_partitions, 0);

    for (int64_t node = 0; node < num_nodes; ++node)
    {
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];

        auto per_partition_increase = std::vector<double>(num_partitions, -INFINITY);
        int64_t chosen_partition = -1;
        bool train_flag = train_mask_data[node];
        int64_t degree_threshold = 25;

        #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
        for (int64_t part = 0; part < num_partitions; ++part)
        {
            auto in_partition_nodes = partition_has_nodes[part];
            int64_t cur_partition_size = partition_size[part];

            int64_t common_neighbor_num = 0;
            for (int64_t pos = st_pos; pos < ed_pos; ++pos)
            {
                auto neighbor = csr_indices_data[pos];
                auto temp_iter = std::lower_bound(in_partition_nodes.begin(), in_partition_nodes.end(),
                                                neighbor);
                if (temp_iter != in_partition_nodes.end() && *temp_iter == neighbor)
                    ++common_neighbor_num;
            }

            per_partition_increase[part] = (double)common_neighbor_num -
                                        alpha*gamma*pow((double)cur_partition_size, gamma-1) -
                                        pow((double)train_partition_size[part], 0.5);
        }

        double max_increase = -INFINITY;
        for (int64_t part = 0; part < num_partitions; ++part)
        {
            double increase = per_partition_increase[part];
            if (increase > max_increase && partition_size[part]+1 <= max_partition_size)
            {
                if (train_flag && train_partition_size[part]+1 > max_train_partition_size)
                    continue;

                max_increase = increase;
                chosen_partition = part;
            }
        }

        partition_assignment[node] = chosen_partition;
        // always insert the largest value, just put in the end position, naturally a sorted vector in terms of node ID
        partition_has_nodes[chosen_partition].push_back(node);
        partition_size[chosen_partition] += 1;
        if (train_flag)
            train_partition_size[chosen_partition] += 1;

        // if (node % 100000 == 0)
            // std::cout << node << std::endl;
    }

    // for debug
    /*for (auto i = partition_has_nodes[0].begin(); i != partition_has_nodes[0].end(); ++i)
        std::cout << *i << " ";
    std::cout << std::endl;*/

    // calcualte edge cut
    auto mapping_data = mapping.data_ptr<int64_t>();
    int64_t cur_homo_cnt = 0;
    for (int64_t node = 0; node < num_nodes; ++node)
    {
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];
        cross_edge_num_data[mapping_data[node]] = ed_pos - st_pos;

        int64_t part_choice_1 = partition_assignment[node];
        for (int64_t pos = st_pos; pos < ed_pos; ++pos)
        {
            if (part_choice_1 == partition_assignment[csr_indices_data[pos]])
            {
                ++cur_homo_cnt;
                --cross_edge_num_data[mapping_data[node]];
            }
        }
    }
    std::cout << "Round 1: edge_cut=" << num_edges-cur_homo_cnt << ", total_edges=" << num_edges
                        << ", ratio=" << (int)((double)(num_edges-cur_homo_cnt)/num_edges*100.0) / 100.0 << std::endl;
    auto homo_cnt = cur_homo_cnt;

    if (patience == -1)
        patience = int((double)num_rounds * 0.2);
    for (int64_t round = 1; round < num_rounds; ++round)
    {
        for (int64_t node = 0; node < num_nodes; ++node)
        {
            int64_t st_pos = csr_indptr_data[node];
            int64_t ed_pos = csr_indptr_data[node+1];

            auto per_partition_increase = std::vector<double>(num_partitions, -INFINITY);

            #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
            for (int64_t part = 0; part < num_partitions; ++part)
            {
                auto in_partition_nodes = partition_has_nodes[part];
                int64_t cur_partition_size = partition_size[part];

                int64_t common_neighbor_num = 0;
                for (int64_t pos = st_pos; pos < ed_pos; ++pos)
                {
                    auto neighbor = csr_indices_data[pos];
                    auto temp_iter = std::lower_bound(in_partition_nodes.begin(), in_partition_nodes.end(),
                                            neighbor);
                    if (temp_iter != in_partition_nodes.end() && *temp_iter == neighbor)
                        ++common_neighbor_num;
                }
                per_partition_increase[part] = (double)common_neighbor_num -
                                        alpha*gamma*pow((double)cur_partition_size, gamma-1) -
                                        pow((double)train_partition_size[part], 0.5);
            }

            double max_increase = -INFINITY;
            int64_t chosen_partition = -1;
            bool train_flag = train_mask_data[node];
            for (int64_t part = 0; part < num_partitions; ++part)
            {
                double increase = per_partition_increase[part];
                if (increase > max_increase && partition_size[part]+1 <= max_partition_size)
                {
                    if (train_flag && train_partition_size[part]+1 > max_train_partition_size)
                        continue;

                    max_increase = increase;
                    chosen_partition = part;
                }
            }

            auto cur_part = partition_assignment[node];
            if (cur_part != chosen_partition)
            {
                --partition_size[cur_part];
                ++partition_size[chosen_partition];
                partition_assignment[node] = chosen_partition;
                if (train_flag)
                {
                    --train_partition_size[cur_part];
                    ++train_partition_size[chosen_partition];
                }
            }
        }

        // calcualte edge cut
        int64_t cur_homo_cnt = 0;
        for (int64_t node = 0; node < num_nodes; ++node)
        {
            int64_t st_pos = csr_indptr_data[node];
            int64_t ed_pos = csr_indptr_data[node+1];
            cross_edge_num_data[mapping_data[node]] = ed_pos - st_pos;

            int64_t part_choice_1 = partition_assignment[node];
            for (int64_t pos = st_pos; pos < ed_pos; ++pos)
            {
                if (part_choice_1 == partition_assignment[csr_indices_data[pos]])
                {
                    cur_homo_cnt += 1;
                    --cross_edge_num_data[mapping_data[node]];
                }
            }
        }
        std::cout << "Round " << round+1 << ": edge_cut=" << num_edges-cur_homo_cnt << ", total_edges=" << num_edges
                        << ", ratio=" << (int)((double)(num_edges-cur_homo_cnt)/num_edges*100.0) / 100.0 << std::endl;

        // maintain information of in-partition nodes
        for (int64_t part = 0; part < num_partitions; ++part)
            partition_has_nodes[part] = std::vector<int64_t>();
        for (int64_t node = 0; node < num_nodes; ++node)
            partition_has_nodes[partition_assignment[node]].push_back(node);

        if (cur_homo_cnt > homo_cnt)
            homo_cnt = cur_homo_cnt;
        else
        {
            --patience;
            if (patience <= 0)
            {
                std::cout << "Early stop at round " << round+1 << std::endl;
                break;
            }
        }
    }

    // save the partitions to disk/SSD
    auto result_data = result.data_ptr<int64_t>();
    auto row_len = result.numel() / num_partitions;
    auto cur_size = std::vector<int64_t>(num_partitions, 0);

    for (int64_t part = 0; part < num_partitions; ++part)
    {
        auto inpart_nodes = partition_has_nodes[part];
        int64_t partition_size = inpart_nodes.size();

        for (int64_t i = 0; i < partition_size; ++i)
        {
            auto node = inpart_nodes[i];
            result_data[part*row_len + cur_size[part]] = node;
            ++cur_size[part];
        }
    }
}


void fennel_bf_partition(torch::Tensor result, torch::Tensor cross_edge_num,
                        torch::Tensor csr_indptr, torch::Tensor csr_indices,
                        int64_t num_partitions, int64_t num_rounds,  int64_t num_train_nodes,
                        torch::Tensor train_mask, int patience, double imbalance_ratio, double gamma, torch::Tensor mapping)
{
    bloom_parameters params;
    // How many elements roughly do we expect to insert?
    params.projected_element_count = int64_t(csr_indptr.numel() - 1 / num_partitions) + 1;
    // Maximum tolerable false positive probability? (0,1)
    params.false_positive_probability = 0.1;
    params.compute_optimal_parameters();

    std::vector<bloom_filter> partition_bf;
    for (int64_t i = 0; i < num_partitions; ++i)
        partition_bf.push_back(bloom_filter(params));

    // fennel partition
    auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
    auto csr_indices_data = csr_indices.data_ptr<int64_t>();

    auto train_mask_data = train_mask.data_ptr<bool>();

    auto cross_edge_num_data = cross_edge_num.data_ptr<int64_t>();

    int64_t num_nodes = csr_indptr.numel() - 1;
    int64_t num_edges = csr_indices.numel();
    double alpha = pow((double)num_partitions, gamma-1) * (double)num_edges / pow((double)num_nodes, gamma);
    double max_partition_size = (double)num_nodes / (double)num_partitions * imbalance_ratio;

    double max_train_partition_size = (double)num_train_nodes / (double)num_partitions * imbalance_ratio;

    std::vector<std::vector<int64_t> > partition_has_nodes;
    for (int64_t i = 0; i < num_partitions; ++i)
        partition_has_nodes.push_back(std::vector<int64_t>());

    auto partition_assignment = std::vector<int64_t>(num_nodes, 0);
    auto partition_size = std::vector<int64_t>(num_partitions, 0);
    auto train_partition_size = std::vector<int64_t>(num_partitions, 0);

    for (int64_t node = 0; node < num_nodes; ++node) {
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];

        auto per_partition_increase = std::vector<double>(num_partitions, -INFINITY);
        int64_t chosen_partition = -1;
        bool train_flag = train_mask_data[node];

        #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
        for (int64_t part = 0; part < num_partitions; ++part)
        {
            int64_t cur_partition_size = partition_size[part];
            int64_t common_neighbor_num = 0;

            for (int64_t pos = st_pos; pos < ed_pos; ++pos)
            {
                auto neighbor = csr_indices_data[pos];
                if (partition_bf[part].contains(neighbor))
                    ++common_neighbor_num;
            }
            per_partition_increase[part] = (double)common_neighbor_num -
                                        alpha*gamma*pow((double)cur_partition_size, gamma-1) -
                                        pow((double)train_partition_size[part], 0.5);
        }

        double max_increase = -INFINITY;
        for (int64_t part = 0; part < num_partitions; ++part)
        {
            double increase = per_partition_increase[part];
            if (increase > max_increase && partition_size[part]+1 <= max_partition_size)
            {
                if (train_flag && train_partition_size[part]+1 > max_train_partition_size)
                    continue;

                max_increase = increase;
                chosen_partition = part;
            }
        }

        partition_assignment[node] = chosen_partition;
        // always insert the largest value, just put in the end position, naturally a sorted vector in terms of node ID
        partition_has_nodes[chosen_partition].push_back(node);
        partition_bf[chosen_partition].insert(node);
        partition_size[chosen_partition] += 1;
        if (train_flag)
            train_partition_size[chosen_partition] += 1;

        // if (node % 100000 == 0)
            // std::cout << node << std::endl;
    }

    // for debug
    /*for (auto i = partition_has_nodes[0].begin(); i != partition_has_nodes[0].end(); ++i)
        std::cout << *i << " ";
    std::cout << std::endl;*/

    // calcualte edge cut
    auto mapping_data = mapping.data_ptr<int64_t>();
    int64_t cur_homo_cnt = 0;
    for (int64_t node = 0; node < num_nodes; ++node)
    {
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];
        cross_edge_num_data[mapping_data[node]] = ed_pos - st_pos;

        int64_t part_choice_1 = partition_assignment[node];
        for (int64_t pos = st_pos; pos < ed_pos; ++pos)
        {
            auto neighbor = csr_indices_data[pos];
            if (neighbor >= num_nodes)
            {
                continue;
            }
            else if (part_choice_1 == partition_assignment[neighbor])
            {
                ++cur_homo_cnt;
                --cross_edge_num_data[mapping_data[node]];
            }
        }
    }
    std::cout << "Round 1: edge_cut=" << num_edges-cur_homo_cnt << ", total_edges=" << num_edges
                        << ", ratio=" << (int)((double)(num_edges-cur_homo_cnt)/num_edges*100.0) / 100.0 << std::endl;

    // save the partitions to disk/SSD
    auto result_data = result.data_ptr<int64_t>();
    auto row_len = result.numel() / num_partitions;
    auto cur_size = std::vector<int64_t>(num_partitions, 0);

    for (int64_t part = 0; part < num_partitions; ++part)
    {
        auto inpart_nodes = partition_has_nodes[part];
        int64_t partition_size = inpart_nodes.size();

        for (int64_t i = 0; i < partition_size; ++i)
        {
            auto node = inpart_nodes[i];
            result_data[part*row_len + cur_size[part]] = node;
            ++cur_size[part];
        }
    }
}


torch::Tensor fetch_split_nodes(torch::Tensor inpart_nodes, torch::Tensor mask)
{
    auto inpart_nodes_data = inpart_nodes.data_ptr<int64_t>();
    auto mask_data = mask.data_ptr<bool>();

    auto train_nid = std::vector<int64_t>();
    int64_t partition_size = inpart_nodes.numel();
    for (int64_t i = 0; i < partition_size; ++i)
    {
        auto node = inpart_nodes_data[i];
        if (mask_data[node])
            train_nid.push_back(node);
    }

    auto out_train_nid = torch::from_blob(train_nid.data(), {train_nid.size()}, inpart_nodes.options()).clone();
    return out_train_nid;
}


std::tuple<torch::Tensor, torch::Tensor> fetch_csr(torch::Tensor csr_indptr, torch::Tensor csr_indices, torch::Tensor inpart_nodes)
{
    auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
    auto csr_indices_data = csr_indices.data_ptr<int64_t>();
    auto inpart_nodes_data = inpart_nodes.data_ptr<int64_t>();

    int64_t partition_size = inpart_nodes.numel();

    auto indptr = std::vector<int64_t>(partition_size+1, 0);
    auto indices = std::vector<int64_t>();

    for (int64_t i = 0; i < partition_size; ++i)
    {
        auto node = inpart_nodes_data[i];
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];

        auto neighbor_count = ed_pos - st_pos;
        indptr[i+1] = indptr[i] + neighbor_count;

        for (int64_t pos = st_pos; pos < ed_pos; ++pos)
        {
            int64_t neighbor = csr_indices_data[pos];
            indices.push_back(neighbor);
        }
    }

    auto out_indptr = torch::from_blob(indptr.data(), {indptr.size()}, inpart_nodes.options()).clone();
    auto out_indices = torch::from_blob(indices.data(), {indices.size()}, inpart_nodes.options()).clone();

    return std::make_tuple(out_indptr, out_indices);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fetch_csr_first_level(torch::Tensor csr_indptr, torch::Tensor csr_indices, torch::Tensor inpart_nodes)
{
    auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
    auto csr_indices_data = csr_indices.data_ptr<int64_t>();
    auto inpart_nodes_data = inpart_nodes.data_ptr<int64_t>();

    int64_t partition_size = inpart_nodes.numel();

    auto indptr = std::vector<int64_t>(partition_size+1, 0);
    auto indices = std::vector<int64_t>();

    std::vector<int64_t> inpart_nodes_vec;
    std::unordered_map<int64_t, int64_t> inpart_nodes_map;
    for (int64_t i = 0; i < partition_size; ++i) {
        int64_t n = inpart_nodes_data[i];
        inpart_nodes_vec.push_back(n);
        inpart_nodes_map[n] = i;
    }

    for (int64_t i = 0; i < partition_size; ++i)
    {
        auto node = inpart_nodes_data[i];
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];

        auto neighbor_count = ed_pos - st_pos;
        indptr[i+1] = indptr[i] + neighbor_count;

        for (int64_t pos = st_pos; pos < ed_pos; ++pos)
        {
            int64_t neighbor = csr_indices_data[pos];
            if (inpart_nodes_map.count(neighbor) == 0) {
                inpart_nodes_map[neighbor] = inpart_nodes_vec.size();
                inpart_nodes_vec.push_back(neighbor);
            }
            indices.push_back(inpart_nodes_map[neighbor]);
        }
    }

    auto out_indptr = torch::from_blob(indptr.data(), {indptr.size()}, inpart_nodes.options()).clone();
    auto out_indices = torch::from_blob(indices.data(), {indices.size()}, inpart_nodes.options()).clone();
    auto out_mapping = torch::from_blob(inpart_nodes_vec.data(), {inpart_nodes_vec.size()}, inpart_nodes.options()).clone();

    return std::make_tuple(out_indptr, out_indices, out_mapping);
}


void fennel_partition_outofcore(torch::Tensor result, torch::Tensor cross_edge_num,
								torch::Tensor csr_indptr, std::string csr_indices_file, int64_t num_edges,
								int64_t num_partitions, int64_t num_rounds, int64_t num_train_nodes,
								torch::Tensor train_mask, int patience, double imbalance_ratio, double gamma, torch::Tensor mapping)
{
	// fennel partition
	auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();

	auto train_mask_data = train_mask.data_ptr<bool>();

	auto cross_edge_num_data = cross_edge_num.data_ptr<int64_t>();

	// open file
	int col_fd = open(csr_indices_file.c_str(), O_RDONLY);

	int64_t num_nodes = csr_indptr.numel() - 1;
	double alpha = pow((double)num_partitions, gamma - 1) * (double)num_edges / pow((double)num_nodes, gamma);
	double max_partition_size = (double)num_nodes / (double)num_partitions * imbalance_ratio;

	double max_train_partition_size = (double)num_train_nodes / (double)num_partitions * imbalance_ratio;
	std::vector<std::vector<int64_t>> partition_has_nodes;
	for (int64_t i = 0; i < num_partitions; ++i)
		partition_has_nodes.push_back(std::vector<int64_t>());

	auto partition_assignment = std::vector<int64_t>(num_nodes, 0);
	auto per_partition_increase = std::vector<double>(num_partitions, 0);
	auto partition_size = std::vector<int64_t>(num_partitions, 0);
	auto train_partition_size = std::vector<int64_t>(num_partitions, 0);
	double io_time = 0, compute_time = 0;
	for (int64_t node = 0; node < num_nodes; ++node)
	{
		int64_t *neighbor_buffer = nullptr;
		int64_t *aligned_neighbor_buffer = nullptr;
		int64_t start_offset = 0;

		int64_t st_pos = csr_indptr_data[node];
		int64_t ed_pos = csr_indptr_data[node + 1];
		auto neighbor_count = ed_pos - st_pos;

		auto io_start = std::chrono::high_resolution_clock::now();
		std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);
		auto io_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> io_duration = io_end - io_start;
		io_time += io_duration.count();

		auto compute_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
		for (int64_t part = 0; part < num_partitions; ++part)
		{
			auto in_partition_nodes = partition_has_nodes[part];
			int64_t cur_partition_size = partition_size[part];

			int64_t common_neighbor_num = 0;
			for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
			{
				auto neighbor = aligned_neighbor_buffer[start_offset + nei_idx];
				auto temp_iter = std::lower_bound(in_partition_nodes.begin(), in_partition_nodes.end(),
												  neighbor);
				if (temp_iter != in_partition_nodes.end() && *temp_iter == neighbor)
					++common_neighbor_num;
			}
			per_partition_increase[part] = (double)common_neighbor_num -
										   alpha * gamma * pow((double)cur_partition_size, gamma - 1) -
										   pow((double)train_partition_size[part], 0.5);
		}

		double max_increase = -INFINITY;
		int64_t chosen_partition = -1;
		bool train_flag = train_mask_data[node];
		for (int64_t part = 0; part < num_partitions; ++part)
		{
			double increase = per_partition_increase[part];
			if (increase > max_increase && partition_size[part] + 1 <= max_partition_size)
			{
				if (train_flag && train_partition_size[part] + 1 > max_train_partition_size)
					continue;

				max_increase = increase;
				chosen_partition = part;
			}
		}

		partition_assignment[node] = chosen_partition;
		// always insert the largest value, just put in the end position, naturally a sorted vector
		partition_has_nodes[chosen_partition].push_back(node);
		partition_size[chosen_partition] += 1;
		if (train_flag)
			train_partition_size[chosen_partition] += 1;

		auto compute_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> compute_duration = compute_end - compute_start;
		compute_time += compute_duration.count();

		free(neighbor_buffer);

		// if (node % 100000 == 0)
			// std::cout << node << std::endl;
	}

	std::cout << "I/O time: " << io_time << "s; Compute time: " << compute_time << "s." << std::endl;

	// for debug
	/*for (auto i = partition_has_nodes[0].begin(); i != partition_has_nodes[0].end(); ++i)
		std::cout << *i << " ";
	std::cout << std::endl;*/

	// calcualte edge cut
    auto mapping_data = mapping.data_ptr<int64_t>();
	int64_t cur_homo_cnt = 0;
	for (int64_t node = 0; node < num_nodes; ++node)
	{
		int64_t *neighbor_buffer = nullptr;
		int64_t *aligned_neighbor_buffer = nullptr;
		int64_t start_offset = 0;

		int64_t st_pos = csr_indptr_data[node];
		int64_t ed_pos = csr_indptr_data[node + 1];
		auto neighbor_count = ed_pos - st_pos;
		cross_edge_num_data[mapping_data[node]] = neighbor_count;
		std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);

		int64_t part_choice_1 = partition_assignment[node];
		for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
		{
			auto neighbor = aligned_neighbor_buffer[start_offset + nei_idx];
			if (part_choice_1 == partition_assignment[neighbor])
			{
				++cur_homo_cnt;
				--cross_edge_num_data[mapping_data[node]];
			}
		}

		free(neighbor_buffer);
	}
	std::cout << "Round 1: edge_cut=" << num_edges - cur_homo_cnt << ", total_edges=" << num_edges
			  << ", ratio=" << (int)((double)(num_edges - cur_homo_cnt) / num_edges * 100.0) / 100.0 << std::endl;
	auto homo_cnt = cur_homo_cnt;

	if (patience == -1)
		patience = int((double)num_rounds * 0.2);
	for (int64_t round = 1; round < num_rounds; ++round)
	{
		for (int64_t node = 0; node < num_nodes; ++node)
		{
			int64_t *neighbor_buffer = nullptr;
			int64_t *aligned_neighbor_buffer = nullptr;
			int64_t start_offset = 0;

			int64_t st_pos = csr_indptr_data[node];
			int64_t ed_pos = csr_indptr_data[node + 1];
			auto neighbor_count = ed_pos - st_pos;
			std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);

            #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
			for (int64_t part = 0; part < num_partitions; ++part)
			{
				auto in_partition_nodes = partition_has_nodes[part];
				int64_t cur_partition_size = partition_size[part];

				int64_t common_neighbor_num = 0;
				for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
				{
					auto neighbor = aligned_neighbor_buffer[start_offset + nei_idx];
					auto temp_iter = std::lower_bound(in_partition_nodes.begin(), in_partition_nodes.end(),
													  neighbor);
					if (temp_iter != in_partition_nodes.end() && *temp_iter == neighbor)
						++common_neighbor_num;
				}
				per_partition_increase[part] = (double)common_neighbor_num -
											   alpha * gamma * pow((double)cur_partition_size, gamma - 1) -
											   pow((double)train_partition_size[part], 0.5);
			}

			double max_increase = -INFINITY;
			int64_t chosen_partition = -1;
			bool train_flag = train_mask_data[node];
			for (int64_t part = 0; part < num_partitions; ++part)
			{
				double increase = per_partition_increase[part];
				if (increase > max_increase && partition_size[part] + 1 <= max_partition_size)
				{
					if (train_flag && train_partition_size[part] + 1 > max_train_partition_size)
						continue;

					max_increase = increase;
					chosen_partition = part;
				}
			}

			auto cur_part = partition_assignment[node];
			if (cur_part != chosen_partition)
			{
				--partition_size[cur_part];
				++partition_size[chosen_partition];
				partition_assignment[node] = chosen_partition;
				if (train_flag)
				{
					--train_partition_size[cur_part];
					++train_partition_size[chosen_partition];
				}
			}

			free(neighbor_buffer);
		}

		// calcualte edge cut
		int64_t cur_homo_cnt = 0;
		for (int64_t node = 0; node < num_nodes; ++node)
		{
			int64_t *neighbor_buffer = nullptr;
			int64_t *aligned_neighbor_buffer = nullptr;
			int64_t start_offset = 0;

			int64_t st_pos = csr_indptr_data[node];
			int64_t ed_pos = csr_indptr_data[node + 1];
			auto neighbor_count = ed_pos - st_pos;
			cross_edge_num_data[mapping_data[node]] = neighbor_count;
			std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);

			int64_t part_choice_1 = partition_assignment[node];
			for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
			{
				auto neighbor = aligned_neighbor_buffer[start_offset + nei_idx];
				if (part_choice_1 == partition_assignment[neighbor])
				{
					cur_homo_cnt += 1;
					--cross_edge_num_data[mapping_data[node]];
				}
			}

			free(neighbor_buffer);
		}
		std::cout << "Round " << round + 1 << ": edge_cut=" << num_edges - cur_homo_cnt << ", total_edges=" << num_edges
				  << ", ratio=" << (int)((double)(num_edges - cur_homo_cnt) / num_edges * 100.0) / 100.0 << std::endl;

		// maintain information of in-partition nodes
		for (int64_t part = 0; part < num_partitions; ++part)
			partition_has_nodes[part] = std::vector<int64_t>();
		for (int64_t node = 0; node < num_nodes; ++node)
			partition_has_nodes[partition_assignment[node]].push_back(node);

		if (cur_homo_cnt > homo_cnt)
			homo_cnt = cur_homo_cnt;
		else
		{
			--patience;
			if (patience <= 0)
			{
				std::cout << "Early stop at round " << round + 1 << std::endl;
				break;
			}
		}

        /*for (int64_t part = 0; part < num_partitions; ++part)
			std::cout << partition_size[part] << " ";
		std::cout << std::endl;*/
	}

	// save the partitions to disk/SSD
	auto result_data = result.data_ptr<int64_t>();
	auto row_len = result.numel() / num_partitions;
	auto cur_size = std::vector<int64_t>(num_partitions, 0);

	for (int64_t part = 0; part < num_partitions; ++part)
	{
		auto inpart_nodes = partition_has_nodes[part];
		int64_t partition_size = inpart_nodes.size();

		for (int64_t i = 0; i < partition_size; ++i)
		{
			auto node = inpart_nodes[i];
			result_data[part * row_len + cur_size[part]] = node;
			++cur_size[part];
		}
	}

	close(col_fd);
}


void fennel_bf_partition_outofcore(torch::Tensor result, torch::Tensor cross_edge_num,
								torch::Tensor csr_indptr, std::string csr_indices_file, int64_t num_edges,
								int64_t num_partitions, int64_t num_rounds, int64_t num_train_nodes,
								torch::Tensor train_mask, int patience, double imbalance_ratio, double gamma, torch::Tensor mapping)
{
	bloom_parameters params;
    // How many elements roughly do we expect to insert?
    params.projected_element_count = int64_t(csr_indptr.numel() - 1 / num_partitions) + 1;
    // Maximum tolerable false positive probability? (0,1)
    params.false_positive_probability = 0.1;
    params.compute_optimal_parameters();

    std::vector<bloom_filter> partition_bf;
    for (int64_t i = 0; i < num_partitions; ++i)
        partition_bf.push_back(bloom_filter(params));

    // fennel partition
	auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();

	auto train_mask_data = train_mask.data_ptr<bool>();

	auto cross_edge_num_data = cross_edge_num.data_ptr<int64_t>();

	// open file
	int col_fd = open(csr_indices_file.c_str(), O_RDONLY);

	int64_t num_nodes = csr_indptr.numel() - 1;
	double alpha = pow((double)num_partitions, gamma - 1) * (double)num_edges / pow((double)num_nodes, gamma);
	double max_partition_size = (double)num_nodes / (double)num_partitions * imbalance_ratio;

	double max_train_partition_size = (double)num_train_nodes / (double)num_partitions * imbalance_ratio;
	std::vector<std::vector<int64_t>> partition_has_nodes;
	for (int64_t i = 0; i < num_partitions; ++i)
		partition_has_nodes.push_back(std::vector<int64_t>());

	auto partition_assignment = std::vector<int64_t>(num_nodes, 0);
	auto per_partition_increase = std::vector<double>(num_partitions, 0);
	auto partition_size = std::vector<int64_t>(num_partitions, 0);
	auto train_partition_size = std::vector<int64_t>(num_partitions, 0);
	double io_time = 0, compute_time = 0;
	for (int64_t node = 0; node < num_nodes; ++node)
	{
		int64_t *neighbor_buffer = nullptr;
		int64_t *aligned_neighbor_buffer = nullptr;
		int64_t start_offset = 0;

		int64_t st_pos = csr_indptr_data[node];
		int64_t ed_pos = csr_indptr_data[node + 1];
		auto neighbor_count = ed_pos - st_pos;

		auto io_start = std::chrono::high_resolution_clock::now();
		std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);
		auto io_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> io_duration = io_end - io_start;
		io_time += io_duration.count();

		auto compute_start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
		for (int64_t part = 0; part < num_partitions; ++part)
		{
			int64_t cur_partition_size = partition_size[part];
            int64_t common_neighbor_num = 0;

            for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
            {
                auto neighbor = aligned_neighbor_buffer[start_offset + nei_idx];
                if (partition_bf[part].contains(neighbor))
                    ++common_neighbor_num;
            }
            per_partition_increase[part] = (double)common_neighbor_num -
                                        alpha*gamma*pow((double)cur_partition_size, gamma-1) -
                                        pow((double)train_partition_size[part], 0.5);
		}

		double max_increase = -INFINITY;
		int64_t chosen_partition = -1;
		bool train_flag = train_mask_data[node];
		for (int64_t part = 0; part < num_partitions; ++part)
		{
			double increase = per_partition_increase[part];
			if (increase > max_increase && partition_size[part] + 1 <= max_partition_size)
			{
				if (train_flag && train_partition_size[part] + 1 > max_train_partition_size)
					continue;

				max_increase = increase;
				chosen_partition = part;
			}
		}

		partition_assignment[node] = chosen_partition;
		// always insert the largest value, just put in the end position, naturally a sorted vector
		partition_has_nodes[chosen_partition].push_back(node);
        partition_bf[chosen_partition].insert(node);
		partition_size[chosen_partition] += 1;
		if (train_flag)
			train_partition_size[chosen_partition] += 1;

		auto compute_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> compute_duration = compute_end - compute_start;
		compute_time += compute_duration.count();

		free(neighbor_buffer);

		// if (node % 100000 == 0)
			// std::cout << node << std::endl;
	}

	std::cout << "I/O time: " << io_time << "s; Compute time: " << compute_time << "s." << std::endl;

	// for debug
	/*for (auto i = partition_has_nodes[0].begin(); i != partition_has_nodes[0].end(); ++i)
		std::cout << *i << " ";
	std::cout << std::endl;*/

	// calcualte edge cut
    auto mapping_data = mapping.data_ptr<int64_t>();
	int64_t cur_homo_cnt = 0;
	for (int64_t node = 0; node < num_nodes; ++node)
	{
		int64_t *neighbor_buffer = nullptr;
		int64_t *aligned_neighbor_buffer = nullptr;
		int64_t start_offset = 0;

		int64_t st_pos = csr_indptr_data[node];
		int64_t ed_pos = csr_indptr_data[node + 1];
		auto neighbor_count = ed_pos - st_pos;
		cross_edge_num_data[mapping_data[node]] = neighbor_count;
		std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);

		int64_t part_choice_1 = partition_assignment[node];
		for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
		{
			auto neighbor = aligned_neighbor_buffer[start_offset + nei_idx];
            if (neighbor >= num_nodes)
            {
                continue;
            }
            else if (part_choice_1 == partition_assignment[neighbor])
            {
                ++cur_homo_cnt;
                --cross_edge_num_data[mapping_data[node]];
            }
		}

		free(neighbor_buffer);
	}
	std::cout << "Round 1: edge_cut=" << num_edges - cur_homo_cnt << ", total_edges=" << num_edges
			  << ", ratio=" << (int)((double)(num_edges - cur_homo_cnt) / num_edges * 100.0) / 100.0 << std::endl;

	// save the partitions to disk/SSD
	auto result_data = result.data_ptr<int64_t>();
	auto row_len = result.numel() / num_partitions;
	auto cur_size = std::vector<int64_t>(num_partitions, 0);

	for (int64_t part = 0; part < num_partitions; ++part)
	{
		auto inpart_nodes = partition_has_nodes[part];
		int64_t partition_size = inpart_nodes.size();

		for (int64_t i = 0; i < partition_size; ++i)
		{
			auto node = inpart_nodes[i];
			result_data[part * row_len + cur_size[part]] = node;
			++cur_size[part];
		}
	}

	close(col_fd);
}


std::tuple<torch::Tensor, torch::Tensor> fetch_csr_outofcore(torch::Tensor csr_indptr, std::string csr_indices_file, torch::Tensor inpart_nodes)
{
    auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
    auto inpart_nodes_data = inpart_nodes.data_ptr<int64_t>();

    int col_fd = open(csr_indices_file.c_str(), O_RDONLY);

    int64_t partition_size = inpart_nodes.numel();

    auto indptr = std::vector<int64_t>(partition_size+1, 0);
    auto indices = std::vector<int64_t>();

    for (int64_t i = 0; i < partition_size; ++i)
    {
        int64_t* neighbor_buffer = nullptr;
        int64_t* aligned_neighbor_buffer = nullptr;
        int64_t start_offset = 0;

        auto node = inpart_nodes_data[i];
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];

        auto neighbor_count = ed_pos - st_pos;
        indptr[i+1] = indptr[i] + neighbor_count;

        std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);

        for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
        {
            auto neighbor = aligned_neighbor_buffer[start_offset+nei_idx];
            indices.push_back(neighbor);
        }

        free(neighbor_buffer);
    }
    close(col_fd);

    auto out_indptr = torch::from_blob(indptr.data(), {indptr.size()}, inpart_nodes.options()).clone();
    auto out_indices = torch::from_blob(indices.data(), {indices.size()}, inpart_nodes.options()).clone();

    return std::make_tuple(out_indptr, out_indices);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fetch_csr_first_level_outofcore(torch::Tensor csr_indptr, std::string csr_indices_file, torch::Tensor inpart_nodes)
{
    auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();

    int col_fd = open(csr_indices_file.c_str(), O_RDONLY);

    auto inpart_nodes_data = inpart_nodes.data_ptr<int64_t>();

    int64_t partition_size = inpart_nodes.numel();

    auto indptr = std::vector<int64_t>(partition_size+1, 0);
    auto indices = std::vector<int64_t>();

    std::vector<int64_t> inpart_nodes_vec;
    std::unordered_map<int64_t, int64_t> inpart_nodes_map;
    for (int64_t i = 0; i < partition_size; ++i) {
        int64_t n = inpart_nodes_data[i];
        inpart_nodes_vec.push_back(n);
        inpart_nodes_map[n] = i;
    }

    for (int64_t i = 0; i < partition_size; ++i)
    {
        int64_t* neighbor_buffer = nullptr;
        int64_t* aligned_neighbor_buffer = nullptr;
        int64_t start_offset = 0;

        auto node = inpart_nodes_data[i];
        int64_t st_pos = csr_indptr_data[node];
        int64_t ed_pos = csr_indptr_data[node+1];

        auto neighbor_count = ed_pos - st_pos;
        indptr[i+1] = indptr[i] + neighbor_count;

        std::tie(neighbor_buffer, aligned_neighbor_buffer, start_offset) = get_neighbors(col_fd, st_pos, neighbor_count);

        for (int64_t nei_idx = 0; nei_idx < neighbor_count; ++nei_idx)
        {
            auto neighbor = aligned_neighbor_buffer[start_offset+nei_idx];
            if (inpart_nodes_map.count(neighbor) == 0) {
                inpart_nodes_map[neighbor] = inpart_nodes_vec.size();
                inpart_nodes_vec.push_back(neighbor);
            }
            indices.push_back(inpart_nodes_map[neighbor]);
        }

        free(neighbor_buffer);
    }
    close(col_fd);

    auto out_indptr = torch::from_blob(indptr.data(), {indptr.size()}, inpart_nodes.options()).clone();
    auto out_indices = torch::from_blob(indices.data(), {indices.size()}, inpart_nodes.options()).clone();
    auto out_mapping = torch::from_blob(inpart_nodes_vec.data(), {inpart_nodes_vec.size()}, inpart_nodes.options()).clone();

    return std::make_tuple(out_indptr, out_indices, out_mapping);
}


PYBIND11_MODULE(partition, m) {
	m.def("fennel_partition", &fennel_partition, "partition the graph by fennel algorithm");
    m.def("fennel_bf_partition", &fennel_bf_partition, "partition the graph by fennel algorithm with Bloom Filter for cardinality estimation");
    m.def("fennel_partition_outofcore", &fennel_partition_outofcore, "partition the graph by fennel algorithm in out-of-core manner");
    m.def("fennel_bf_partition_outofcore", &fennel_bf_partition_outofcore, "partition the graph by fennel algorithm in out-of-core manner");
    m.def("fetch_csr", &fetch_csr, "fetch csr");
    m.def("fetch_csr_first_level", &fetch_csr_first_level, "fetch csr first level");
    m.def("fetch_csr_first_level_outofcore", &fetch_csr_first_level_outofcore, "fetch csr first level");
    m.def("fetch_csr_outofcore", &fetch_csr_outofcore, "fetch csr out-of-core");
    m.def("fetch_split_nodes", &fetch_split_nodes, "fetch split nodes");
}
