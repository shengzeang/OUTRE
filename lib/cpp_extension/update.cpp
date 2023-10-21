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


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> cache_check(torch::Tensor n_id, torch::Tensor emb_cache_table, torch::Tensor true_emb_cache_table, torch::Tensor emb_cache_stale, int64_t n_iter)
{
	auto n_id_data = n_id.data_ptr<int64_t>();
    auto emb_cache_table_data = emb_cache_table.data_ptr<int32_t>();
	auto true_emb_cache_table_data = true_emb_cache_table.data_ptr<int32_t>();
	auto emb_cache_stale_data = emb_cache_stale.data_ptr<int32_t>();

	std::vector<int64_t> push_batch_id, pull_batch_id;
	std::vector<int64_t> push_global_id, pull_global_id;

	auto batch_size = n_id.numel();
	for (int64_t i = 0; i < batch_size; ++i)
	{
		auto node = n_id_data[i];
		auto entry_value = emb_cache_table_data[node];
		auto pos = true_emb_cache_table_data[node];
        if (entry_value == -5)
        {
            push_batch_id.push_back(i);
            push_global_id.push_back(node);
            emb_cache_stale_data[pos] = n_iter;
        }
		else if (entry_value == -1)
		{
			if (abs(n_iter - emb_cache_stale_data[pos]) >= atoi(getenv("STALENESS")))
			{
				push_batch_id.push_back(i);
				push_global_id.push_back(node);
				emb_cache_stale_data[pos] = n_iter;
			}
			else
			{
				pull_batch_id.push_back(i);
				pull_global_id.push_back(node);
			}
		}
	}

	auto out_push_batch_id = torch::from_blob(push_batch_id.data(), {push_batch_id.size()}, n_id.options()).clone();
	auto out_push_global_id = torch::from_blob(push_global_id.data(), {push_global_id.size()}, n_id.options()).clone();
	auto out_pull_batch_id = torch::from_blob(pull_batch_id.data(), {pull_batch_id.size()}, n_id.options()).clone();
	auto out_pull_global_id = torch::from_blob(pull_global_id.data(), {pull_global_id.size()}, n_id.options()).clone();

	return std::make_tuple(out_push_batch_id, out_push_global_id, out_pull_batch_id, out_pull_global_id);
}


void update_feat_cache(torch::Tensor n_id, torch::Tensor batch_inputs, torch::Tensor cache, torch::Tensor address_table, torch::Tensor true_address_table, int64_t num_features)
{
	auto cache_data = cache.data_ptr<float>();
    auto address_table_data = address_table.data_ptr<int32_t>();
	auto true_address_table_data = true_address_table.data_ptr<int32_t>();
    auto batch_inputs_data = batch_inputs.data_ptr<float>();
	auto n_id_data = n_id.data_ptr<int64_t>();

	int64_t num_idx = n_id.numel();
    int64_t feature_size = num_features*sizeof(float);

	#pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; ++n) {
		int64_t global_id = n_id_data[n];
		int32_t pos = true_address_table_data[global_id];
		if (pos >= 0 && address_table_data[global_id] == -1)
		{
			memcpy(cache_data+num_features*pos, batch_inputs_data+num_features*n, feature_size);
			address_table_data[global_id] = pos;
		}
    }

	return;
}


PYBIND11_MODULE(update, m) {
	m.def("cache_check", &cache_check, "cache check, determine which to push and pull");
	m.def("update_feat_cache", &update_feat_cache, "update feature cache during training");
}


