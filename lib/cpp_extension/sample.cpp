#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <errno.h>
#include <cstring>
#include <inttypes.h>
#include <omp.h>
#include <iostream>
#include <math.h>
#define ALIGNMENT 4096

// return start index of buffer
int64_t load_neighbors_into_buffer(int col_fd, int64_t row_start, int64_t row_count, int64_t* buffer){
    int64_t size = (row_count*sizeof(int64_t) + 2*ALIGNMENT)&(long)~(ALIGNMENT-1);
    int64_t offset = row_start*sizeof(int64_t);
    int64_t aligned_offset = offset&(long)~(ALIGNMENT-1);

    if(pread(col_fd, buffer, size, aligned_offset) == -1){
        fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }

    return (offset-aligned_offset)/sizeof(int64_t);
}

std::tuple<int64_t*, int64_t*, int64_t> get_new_neighbor_buffer(int64_t row_count){
    int64_t size = (row_count*sizeof(int64_t) + 3*ALIGNMENT)&(long)~(ALIGNMENT-1);
    int64_t* neighbor_buffer = (int64_t*)malloc(size + ALIGNMENT);
    int64_t* aligned_neighbor_buffer = (int64_t*)(((long)neighbor_buffer+(long)ALIGNMENT)&(long)~(ALIGNMENT-1));

    return std::make_tuple(neighbor_buffer, aligned_neighbor_buffer, size/sizeof(int64_t));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj_prune_rm_repeat(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                  torch::Tensor pre_row_ptr, torch::Tensor pre_cols, int64_t start_idx,
                  torch::Tensor cache, torch::Tensor cache_table,
                  torch::Tensor inpart_idx, torch::Tensor csr_indptr, torch::Tensor csr_indices,
                  torch::Tensor emb_cache_table, torch::Tensor true_emb_cache_table, torch::Tensor emb_cache_stale,
                  int64_t batch_count, int64_t layer, int64_t num_neighbors, bool replace) {

  omp_set_num_threads(1);

  srand(time(NULL) + 1000 * getpid()); // Initialize random seed.

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();
  auto cache_data = cache.data_ptr<int64_t>();
  auto cache_table_data = cache_table.data_ptr<int64_t>();
  auto emb_cache_table_data = emb_cache_table.data_ptr<int32_t>();
  auto emb_cache_stale_data = emb_cache_stale.data_ptr<int32_t>();
  auto true_emb_cache_table_data = true_emb_cache_table.data_ptr<int32_t>();

  auto inpart_idx_data = inpart_idx.data_ptr<int64_t>();
  auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
  auto csr_indices_data = csr_indices.data_ptr<int64_t>();

  auto pre_row_ptr_data = pre_row_ptr.data_ptr<int64_t>();
  auto pre_cols_data = pre_cols.data_ptr<int64_t>();

  auto out_rowptr = torch::empty(idx.numel() + 1, idx.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  std::vector<std::vector<int64_t> > cols; // col, e_id
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;

  std::unordered_set<int64_t> inpart_idx_set;
  std::unordered_map<int64_t, int64_t> inpart_idx_map;
  int64_t i;
  for (int64_t n = 0; n < inpart_idx.numel(); n++) {
    i = inpart_idx_data[n];
    inpart_idx_set.insert(i);
    inpart_idx_map[i] = n;
  }

  for (int64_t n = 0; n < idx.numel(); n++) {
    i = idx_data[n];
    cols.push_back(std::vector<int64_t>());
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  int64_t n, c, e, row_start, row_end, row_count;
  int64_t start_offset;
  int64_t cache_entry;
  int64_t indptr_entry;

  int64_t hit_num_1 = 0, hit_num_2 = 0;

  if (num_neighbors < 0) { // Full neighbor sampling ======================================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      if (cache_entry >= 0){
          row_count = cache_data[cache_entry];
          for (int64_t j = 0; j < row_count; j++){
            e = cache_entry + 1 + j;
            c = cache_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;

          for (int64_t j = 0; j < row_count; j++) {
            e = row_start + j;
            c = col_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }
 ///
  else if (replace) { // Sample with replacement ===============================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      if (cache_entry >= 0){
          row_count = cache_data[cache_entry];
          for (int64_t j = 0; j < num_neighbors; j++){
            e = cache_entry + 1 + rand() % row_count;
            c = cache_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;
          if (row_count > 0) {
            for (int64_t j = 0; j < num_neighbors; j++) {
              e = row_start + rand() % row_count;
              c = col_data[e];

              if (n_id_map.count(c) == 0) {
                n_id_map[c] = n_ids.size();
                n_ids.push_back(c);
              }
              cols[i].push_back(n_id_map[c]);
            }
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }

  } else { // Sample without replacement via Robert Floyd algorithm ============

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];

      if (layer == 1)
      {
        auto pos = true_emb_cache_table_data[n];
        if (emb_cache_table_data[n] == -2)
        {
          emb_cache_table_data[n] = -5;
          emb_cache_stale_data[pos] = batch_count;
        }
        else if (emb_cache_table_data[n] != -3)
        {
          emb_cache_table_data[n] = -1;
          if (abs(batch_count - emb_cache_stale_data[pos]) < atoi(getenv("STALENESS")))
          {
            out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
            continue;
          }
          else
          {
            emb_cache_table_data[n] = -5;
            emb_cache_stale_data[pos] = batch_count;
          }
        }
      }
      else if (layer > 1 && emb_cache_table_data[n] == -1)
      {
        out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
        continue;
      }

      cache_entry = cache_table_data[n];
      auto n_cnt = inpart_idx_set.count(n);
      if (n_cnt > 1)
          assert(false);

      if (i < start_idx){
          row_start = pre_row_ptr_data[i];
          row_end = pre_row_ptr_data[i+1];
          for (int64_t j = row_start; j < row_end; j++)
          {
              c = pre_cols_data[j];
              cols[i].push_back(c);
          }
      }
      else if (cache_entry >= 0){
          hit_num_2 += 1;

          // cache hit
          row_count = cache_data[cache_entry];
          if (row_count > 0){
              std::unordered_set<int64_t> perm;
              if (row_count <= num_neighbors) {
                for (int64_t j = 0; j < row_count; j++)
                  perm.insert(j);
              }
              else {
                for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
                  if (!perm.insert(rand() % j).second)
                    perm.insert(j);
                }
              }

              for (const int64_t &p : perm) {
                e = cache_entry + 1 + p;
                c = cache_data[e];

                if (n_id_map.count(c) == 0) {
                  n_id_map[c] = n_ids.size();
                  n_ids.push_back(c);
                }
                cols[i].push_back(n_id_map[c]);
                if (layer >= 1 && emb_cache_table_data[c] == -1)
                {
                  emb_cache_table_data[c] = -4;
                }
              }
          }
      }
      else if (n_cnt > 0){
          hit_num_1 += 1;

          auto pos = inpart_idx_map[n];

          indptr_entry = csr_indptr_data[pos];
          row_count = csr_indptr_data[pos+1] - indptr_entry;

          std::unordered_set<int64_t> perm;
          if (row_count <= num_neighbors) {
            for (int64_t j = 0; j < row_count; j++)
              perm.insert(j);
          }
          else {
            for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
              if (!perm.insert(rand() % j).second)
                perm.insert(j);
            }
          }

          for (const int64_t &p : perm) {
            e = indptr_entry + p;
            c = csr_indices_data[e];

            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
            if (layer >= 1 && emb_cache_table_data[c] == -1)
            {
              emb_cache_table_data[c] = -4;
            }
          }
      }
      else {
          // cache miss
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;

          if (row_count > 0){
              std::unordered_set<int64_t> perm;
              if (row_count <= num_neighbors) {
                for (int64_t j = 0; j < row_count; j++)
                  perm.insert(j);
              }
              else {
                for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
                  if (!perm.insert(rand() % j).second)
                    perm.insert(j);
                }
              }

              for (const int64_t &p : perm) {
                e = row_start + p;
                c = col_data[e];

                if (n_id_map.count(c) == 0) {
                  n_id_map[c] = n_ids.size();
                  n_ids.push_back(c);
                }
                cols[i].push_back(n_id_map[c]);
                if (layer >= 1 && emb_cache_table_data[c] == -1)
                {
                  emb_cache_table_data[c] = -4;
                }
              }
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }
 ///

  int64_t N = n_ids.size();
  auto out_n_id = torch::from_blob(n_ids.data(), {N}, idx.options()).clone();

  int64_t E = out_rowptr_data[idx.numel()];
  auto out_col = torch::empty(E, idx.options());
  auto out_col_data = out_col.data_ptr<int64_t>();

  i = 0;
  for (std::vector<int64_t> &col_vec : cols) {
    std::sort(col_vec.begin(), col_vec.end(),
              [](const int64_t &a,
                 const int64_t &b) -> bool {
                return a < b;
              });
    for (const int64_t &value : col_vec) {
      out_col_data[i] = value;
      i += 1;
    }
  }

  return std::make_tuple(out_rowptr, out_col, out_n_id);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj_profiling(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                  torch::Tensor cache, torch::Tensor cache_table,
                  torch::Tensor inpart_idx, torch::Tensor csr_indptr, torch::Tensor csr_indices,
                  int64_t num_neighbors, bool replace) {

  omp_set_num_threads(1);

  srand(time(NULL) + 1000 * getpid()); // Initialize random seed.

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();
  auto cache_data = cache.data_ptr<int64_t>();
  auto cache_table_data = cache_table.data_ptr<int64_t>();

  auto inpart_idx_data = inpart_idx.data_ptr<int64_t>();
  auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
  auto csr_indices_data = csr_indices.data_ptr<int64_t>();

  auto out_rowptr = torch::empty(idx.numel() + 1, idx.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  auto out_rowptr_neigh = torch::empty(idx.numel() + 1, idx.options());
  auto out_rowptr_neigh_data = out_rowptr_neigh.data_ptr<int64_t>();
  out_rowptr_neigh_data[0] = 0;

  std::vector<std::vector<int64_t> > cols; // col, e_id
  std::vector<std::vector<int64_t> > cols_neigh; // col, e_id
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;

  std::unordered_set<int64_t> inpart_idx_set;
  std::unordered_map<int64_t, int64_t> inpart_idx_map;
  int64_t i;
  for (int64_t n = 0; n < inpart_idx.numel(); n++) {
    i = inpart_idx_data[n];
    inpart_idx_set.insert(i);
    inpart_idx_map[i] = n;
  }

  for (int64_t n = 0; n < idx.numel(); n++) {
    i = idx_data[n];
    cols.push_back(std::vector<int64_t>());
    cols_neigh.push_back(std::vector<int64_t>());
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  int64_t n, c, e, row_start, row_end, row_count;
  int64_t start_offset;
  int64_t cache_entry;
  int64_t indptr_entry;

  int64_t hit_num_1 = 0, hit_num_2 = 0;

  if (num_neighbors < 0) { // Full neighbor sampling ======================================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      if (cache_entry >= 0){
          row_count = cache_data[cache_entry];
          for (int64_t j = 0; j < row_count; j++){
            e = cache_entry + 1 + j;
            c = cache_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;

          for (int64_t j = 0; j < row_count; j++) {
            e = row_start + j;
            c = col_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }
 ///
  else if (replace) { // Sample with replacement ===============================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      if (cache_entry >= 0){
          row_count = cache_data[cache_entry];
          for (int64_t j = 0; j < num_neighbors; j++){
            e = cache_entry + 1 + rand() % row_count;
            c = cache_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;

          if (row_count > 0) {
            for (int64_t j = 0; j < num_neighbors; j++) {
              e = row_start + rand() % row_count;
              c = col_data[e];

              if (n_id_map.count(c) == 0) {
                n_id_map[c] = n_ids.size();
                n_ids.push_back(c);
              }
              cols[i].push_back(n_id_map[c]);
            }
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }

  } else { // Sample without replacement via Robert Floyd algorithm ============

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      auto n_cnt = inpart_idx_set.count(n);
      if (n_cnt > 1)
          assert(false);

      if (cache_entry >= 0){
          hit_num_2 += 1;

          // cache hit
          row_count = cache_data[cache_entry];
          if (row_count > 0){
              std::unordered_set<int64_t> perm;
              if (row_count <= num_neighbors) {
                for (int64_t j = 0; j < row_count; j++)
                  perm.insert(j);
              }
              else {
                for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
                  if (!perm.insert(rand() % j).second)
                    perm.insert(j);
                }
              }

              for (const int64_t &p : perm) {
                e = cache_entry + 1 + p;
                c = cache_data[e];

                if (n_id_map.count(c) == 0) {
                  n_id_map[c] = n_ids.size();
                  n_ids.push_back(c);
                  cols_neigh[i].push_back(n_id_map[c]);
                }
                cols[i].push_back(n_id_map[c]);
              }
          }
      }
      else if (n_cnt > 0){
          hit_num_1 += 1;

          auto pos = inpart_idx_map[n];

          indptr_entry = csr_indptr_data[pos];
          row_count = csr_indptr_data[pos+1] - indptr_entry;

          std::unordered_set<int64_t> perm;
          if (row_count <= num_neighbors) {
            for (int64_t j = 0; j < row_count; j++)
              perm.insert(j);
          }
          else {
            for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
              if (!perm.insert(rand() % j).second)
                perm.insert(j);
            }
          }

          for (const int64_t &p : perm) {
            e = indptr_entry + p;
            c = csr_indices_data[e];

            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
              cols_neigh[i].push_back(n_id_map[c]);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          // cache miss
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;
          if (row_count > 0){
              std::unordered_set<int64_t> perm;
              if (row_count <= num_neighbors) {
                for (int64_t j = 0; j < row_count; j++)
                  perm.insert(j);
              }
              else {
                for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
                  if (!perm.insert(rand() % j).second)
                    perm.insert(j);
                }
              }

              for (const int64_t &p : perm) {
                e = row_start + p;
                c = col_data[e];

                if (n_id_map.count(c) == 0) {
                  n_id_map[c] = n_ids.size();
                  n_ids.push_back(c);
                  cols_neigh[i].push_back(n_id_map[c]);
                }
                cols[i].push_back(n_id_map[c]);
              }
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
      out_rowptr_neigh_data[i + 1] = out_rowptr_neigh_data[i] + cols_neigh[i].size();
    }
  }
 ///

  int64_t N = n_ids.size();
  auto out_n_id = torch::from_blob(n_ids.data(), {N}, idx.options()).clone();

  int64_t E = out_rowptr_data[idx.numel()];
  auto out_col = torch::empty(E, idx.options());
  auto out_col_data = out_col.data_ptr<int64_t>();
  i = 0;
  for (std::vector<int64_t> &col_vec : cols) {
    std::sort(col_vec.begin(), col_vec.end(),
              [](const int64_t &a,
                 const int64_t &b) -> bool {
                return a < b;
              });
    for (const int64_t &value : col_vec) {
      out_col_data[i] = value;
      i += 1;
    }
  }

  int64_t E_neigh = out_rowptr_neigh_data[idx.numel()];
  auto out_col_neigh = torch::empty(E_neigh, idx.options());
  auto out_col_neigh_data = out_col_neigh.data_ptr<int64_t>();
  i = 0;
  for (std::vector<int64_t> &col_neigh_vec : cols_neigh) {
    std::sort(col_neigh_vec.begin(), col_neigh_vec.end(),
              [](const int64_t &a,
                 const int64_t &b) -> bool {
                return a < b;
              });
    for (const int64_t &value : col_neigh_vec) {
      out_col_neigh_data[i] = value;
      i += 1;
    }
  }

  return std::make_tuple(out_rowptr, out_col, out_n_id, out_rowptr_neigh, out_col_neigh);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                  torch::Tensor cache, torch::Tensor cache_table,
                  torch::Tensor inpart_idx, torch::Tensor csr_indptr, torch::Tensor csr_indices,
                  int64_t num_neighbors, bool replace) {

  omp_set_num_threads(1);

  srand(time(NULL) + 1000 * getpid()); // Initialize random seed.

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();
  auto cache_data = cache.data_ptr<int64_t>();
  auto cache_table_data = cache_table.data_ptr<int64_t>();

  auto inpart_idx_data = inpart_idx.data_ptr<int64_t>();
  auto csr_indptr_data = csr_indptr.data_ptr<int64_t>();
  auto csr_indices_data = csr_indices.data_ptr<int64_t>();

  auto out_rowptr = torch::empty(idx.numel() + 1, idx.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  std::vector<std::vector<int64_t> > cols; // col, e_id
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;

  std::unordered_set<int64_t> inpart_idx_set;
  std::unordered_map<int64_t, int64_t> inpart_idx_map;
  int64_t i;
  for (int64_t n = 0; n < inpart_idx.numel(); n++) {
    i = inpart_idx_data[n];
    inpart_idx_set.insert(i);
    inpart_idx_map[i] = n;
  }

  for (int64_t n = 0; n < idx.numel(); n++) {
    i = idx_data[n];
    cols.push_back(std::vector<int64_t>());
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  int64_t n, c, e, row_start, row_end, row_count;
  int64_t start_offset;
  int64_t cache_entry;
  int64_t indptr_entry;

  int64_t hit_num_1 = 0, hit_num_2 = 0;

  if (num_neighbors < 0) { // Full neighbor sampling ======================================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      if (cache_entry >= 0){
          row_count = cache_data[cache_entry];
          for (int64_t j = 0; j < row_count; j++){
            e = cache_entry + 1 + j;
            c = cache_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;

          for (int64_t j = 0; j < row_count; j++) {
            e = row_start + j;
            c = col_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }
 ///
  else if (replace) { // Sample with replacement ===============================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      if (cache_entry >= 0){
          row_count = cache_data[cache_entry];
          for (int64_t j = 0; j < num_neighbors; j++){
            e = cache_entry + 1 + rand() % row_count;
            c = cache_data[e];

            // if c does not exist in n_id_map
            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;

          if (row_count > 0) {
            for (int64_t j = 0; j < num_neighbors; j++) {
              e = row_start + rand() % row_count;
              c = col_data[e];

              if (n_id_map.count(c) == 0) {
                n_id_map[c] = n_ids.size();
                n_ids.push_back(c);
              }
              cols[i].push_back(n_id_map[c]);
            }
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }

  } else { // Sample without replacement via Robert Floyd algorithm ============

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      cache_entry = cache_table_data[n];
      auto n_cnt = inpart_idx_set.count(n);
      if (n_cnt > 1)
          assert(false);

      if (cache_entry >= 0){
          hit_num_2 += 1;

          // cache hit
          row_count = cache_data[cache_entry];

          if (row_count > 0){
              std::unordered_set<int64_t> perm;
              if (row_count <= num_neighbors) {
                for (int64_t j = 0; j < row_count; j++)
                  perm.insert(j);
              }
              else {
                for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
                  if (!perm.insert(rand() % j).second)
                    perm.insert(j);
                }
              }

              for (const int64_t &p : perm) {
                e = cache_entry + 1 + p;
                c = cache_data[e];

                if (n_id_map.count(c) == 0) {
                  n_id_map[c] = n_ids.size();
                  n_ids.push_back(c);
                }
                cols[i].push_back(n_id_map[c]);
              }
          }
      }
      else if (n_cnt > 0){
          hit_num_1 += 1;

          auto pos = inpart_idx_map[n];

          indptr_entry = csr_indptr_data[pos];
          row_count = csr_indptr_data[pos+1] - indptr_entry;

          std::unordered_set<int64_t> perm;
          if (row_count <= num_neighbors) {
            for (int64_t j = 0; j < row_count; j++)
              perm.insert(j);
          }
          else {
            for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
              if (!perm.insert(rand() % j).second)
                perm.insert(j);
            }
          }

          for (const int64_t &p : perm) {
            e = indptr_entry + p;
            c = csr_indices_data[e];

            if (n_id_map.count(c) == 0) {
              n_id_map[c] = n_ids.size();
              n_ids.push_back(c);
            }
            cols[i].push_back(n_id_map[c]);
          }
      }
      else {
          // cache miss
          row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
          row_count = row_end - row_start;

          if (row_count > 0){
              std::unordered_set<int64_t> perm;
              if (row_count <= num_neighbors) {
                for (int64_t j = 0; j < row_count; j++)
                  perm.insert(j);
              }
              else {
                for (int64_t j = row_count - num_neighbors; j < row_count; j++) {
                  if (!perm.insert(rand() % j).second)
                    perm.insert(j);
                }
              }

              for (const int64_t &p : perm) {
                e = row_start + p;
                c = col_data[e];

                if (n_id_map.count(c) == 0) {
                  n_id_map[c] = n_ids.size();
                  n_ids.push_back(c);
                }
                cols[i].push_back(n_id_map[c]);
              }
          }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }
  }
 ///

  int64_t N = n_ids.size();
  auto out_n_id = torch::from_blob(n_ids.data(), {N}, idx.options()).clone();

  int64_t E = out_rowptr_data[idx.numel()];
  auto out_col = torch::empty(E, idx.options());
  auto out_col_data = out_col.data_ptr<int64_t>();

  i = 0;
  for (std::vector<int64_t> &col_vec : cols) {
    std::sort(col_vec.begin(), col_vec.end(),
              [](const int64_t &a,
                 const int64_t &b) -> bool {
                return a < b;
              });
    for (const int64_t &value : col_vec) {
      out_col_data[i] = value;
      i += 1;
    }
  }

  return std::make_tuple(out_rowptr, out_col, out_n_id);
}


torch::Tensor
get_neighbors(torch::Tensor rowptr, std::string col_file, torch::Tensor idx) {

  // open files
  int col_fd = open(col_file.c_str(), O_RDONLY | O_DIRECT);

  // prepare buffer
  int64_t neighbor_buffer_size = 1<<15;
  int64_t* neighbor_buffer = (int64_t*)malloc(neighbor_buffer_size*sizeof(int64_t) + 2*ALIGNMENT);
  int64_t* aligned_neighbor_buffer = (int64_t*)(((long)neighbor_buffer+(long)ALIGNMENT)&(long)~(ALIGNMENT-1));

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>()[0];

  std::vector<int64_t> n_ids;

  int64_t i;

  int64_t n, c, e, row_start, row_end, row_count;
  int64_t start_offset;

  row_start = rowptr_data[idx_data], row_end = rowptr_data[idx_data + 1];
  row_count = row_end - row_start;

  if (row_count > neighbor_buffer_size){
      free(neighbor_buffer);
      std::tie(neighbor_buffer, aligned_neighbor_buffer, neighbor_buffer_size) = get_new_neighbor_buffer(row_count);
  }

  start_offset = load_neighbors_into_buffer(col_fd,  row_start, row_count, aligned_neighbor_buffer);
  for (int64_t j = 0; j < row_count; j++) {
    e = start_offset + j;
    c = aligned_neighbor_buffer[e];
    n_ids.push_back(c);
  }

  int64_t N = n_ids.size();
  auto out_n_id = torch::from_blob(n_ids.data(), {N}, idx.options()).clone();

  free(neighbor_buffer);
  close(col_fd);
  return out_n_id;
}

void fill_neighbor_cache(torch::Tensor cache, torch::Tensor rowptr, std::string col,
                torch::Tensor cached_idx, torch::Tensor cache_table, int64_t num_entries) {

    int64_t* cached_idx_data = cached_idx.data_ptr<int64_t>();
    int64_t* cache_table_data = cache_table.data_ptr<int64_t>();
    int64_t* cache_data = cache.data_ptr<int64_t>();

    #pragma omp parallel for num_threads(atoi(getenv("NUM_THREADS")))
    for (int64_t n=0; n<num_entries; n++){
        int64_t idx = cached_idx_data[n];
        auto neighbors = get_neighbors(rowptr, col, cached_idx[n]);
        int64_t num_neighbors = neighbors.numel();

        int64_t position = cache_table_data[idx];

        // cache update
        cache_data[position] = num_neighbors;
        memcpy(cache_data+position+1, (int64_t*)neighbors.data_ptr(), num_neighbors*sizeof(int64_t));
    }

    return;
}

PYBIND11_MODULE(sample, m) {
    m.def("sample_adj", &sample_adj, "sample adj using csr");
    m.def("sample_adj_profiling", &sample_adj_profiling, "sample adj using csr for profiling during preprocessing");
    m.def("sample_adj_prune_rm_repeat", &sample_adj_prune_rm_repeat, "sample adj using csr");
    m.def("fill_neighbor_cache", &fill_neighbor_cache, "fetch neighbors of given indices into the cache and set the cache table");
}

