#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/utils/ParamUtils.h>
#include <omp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace std {
template <>
struct hash<std::vector<int64_t>> {
  size_t operator()(const std::vector<int64_t>& key) const {
    size_t total = key.size();
    size_t sum = 0;
    if (total < 64) {
      for (size_t i = 0; i < total; i++) {
        sum += key[i] << i;
      }
    } else {
      size_t batch = total / 64;
      size_t remain = total % 64;
      for (size_t bs = 0; bs < batch; bs++) {
        for (size_t i = 0; i < 64; i++) {
          sum += key[bs * 64 + i] << i;
        }
      }
      for (size_t i = 0; i < remain; i++) {
        sum += key[batch * 64 + i] << i;
      }
    }
    return sum;
  }
};

} // namespace std

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

Tensor mkldnn_graph_sdpa_pattern(
    const int uniqueID,
    const Tensor& key,
    const Tensor& query,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& attn_mask) {
  TORCH_CHECK(
      false,
      "mkldnn_graph_sdpa_pattern: ATen not compiled with MKLDNN support");
}

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/Utils.h>
#include <c10/util/irange.h>
#include <oneapi/dnnl/dnnl_graph.hpp>

namespace at {
namespace native {

using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

namespace {

#define ONEDNN_GRAPH_SDPA_PATTERN_5 0

using RunArg = dnnl::graph::tensor;
using RunArgs = std::vector<RunArg>;
using LogicalTensors = std::vector<logical_tensor>;

struct cp_entry {
  partition partition_;
  compiled_partition cp_;
  RunArgs constantInputTensors_;
  RunArgs inputLLGATensors_;
  RunArgs outputLLGATensors_;
  LogicalTensors inputLogicalTensors_;
  LogicalTensors outputLogicalTensors_;
};

std::once_flag init_sdpa_pattern_5_;

static thread_local std::unordered_map<int32_t, dnnl::graph::partition>
    partition_map_;
using key_value_pair_t = std::pair<std::vector<int64_t>, cp_entry>;
using list_iterator_t = std::list<key_value_pair_t>::iterator;
static thread_local std::list<key_value_pair_t> cache_items_list_;
static thread_local std::unordered_map<std::vector<int64_t>, list_iterator_t>
    cache_items_map_;
static thread_local int capacity_ = 75000;

compiled_partition compile_partition(
    const partition& partition,
    const std::vector<logical_tensor>& inputs,
    const std::vector<logical_tensor>& outputs) {
  compiled_partition compilation;
  compilation =
      partition.compile(inputs, outputs, onednn_graph::Engine::getEngine());
  return compilation;
}

/*
   (f32/bf16)[Query]     [Key](f32/bf16)
              \     /
               MatMul
                 |
            Divide  [attn mask]
                 |  /
                 | /
                Add
                 |
              Softmax    [Value](f32/bf16)
                    \     /
                     MatMul
                       |
                    [output](f32/bf16)
*/
void create_graph_sdpa_pattern_5() {
  std::call_once(init_sdpa_pattern_5_, [&]() {
    dnnl::graph::graph g{dnnl::graph::engine::kind::cpu};

    auto DTYPE = data_type::f32;
    size_t op_idx = 0;
    size_t logical_tensor_id = 0;

    logical_tensor q_trans_src_desc(logical_tensor_id++, DTYPE);

    logical_tensor k_trans_src_desc(logical_tensor_id++, DTYPE);

    logical_tensor v_trans_src_desc(logical_tensor_id++, DTYPE);

    logical_tensor matmul_qk_dst_desc{logical_tensor_id++, DTYPE};
    op matmul_qk(
        op_idx++,
        op::kind::MatMul,
        {q_trans_src_desc, k_trans_src_desc},
        {matmul_qk_dst_desc},
        "matmul_qk");

    logical_tensor fscore_scale_desc = {logical_tensor_id++, DTYPE};
    logical_tensor fscore_div_dst_desc = {logical_tensor_id++, DTYPE};
    op fscore_div(
        op_idx++,
        op::kind::Divide,
        {matmul_qk_dst_desc, fscore_scale_desc},
        {fscore_div_dst_desc},
        "fscore_div");

    logical_tensor attension_mask_desc = {logical_tensor_id++, DTYPE};
    logical_tensor fscore_add_dst_desc = {logical_tensor_id++, DTYPE};
    op fscore_add(
        op_idx++,
        op::kind::Add,
        {fscore_div_dst_desc, attension_mask_desc},
        {fscore_add_dst_desc},
        "fscore_add");

    logical_tensor softmax_out_dst_desc = {logical_tensor_id++, DTYPE};
    op softmax_out(
        op_idx++,
        op::kind::SoftMax,
        {fscore_add_dst_desc},
        {softmax_out_dst_desc},
        "softmax_out");
    softmax_out.set_attr<dim>(op::attr::axis, -1);

    logical_tensor matmul_v_dst_desc{logical_tensor_id++, DTYPE};
    op matmul_v(
        op_idx++,
        op::kind::MatMul,
        {softmax_out_dst_desc, v_trans_src_desc},
        {matmul_v_dst_desc},
        "matmul_value");

    g.add_op(matmul_qk);
    g.add_op(fscore_div);
    g.add_op(fscore_add);
    g.add_op(softmax_out);
    g.add_op(matmul_v);

    g.finalize();
    auto partitions = g.get_partitions();
    auto partition = partitions[0];
    TORCH_CHECK(
        (partitions.size() == 1) && partition.is_supported(),
        " only one fusion group allowed");
    partition_map_[ONEDNN_GRAPH_SDPA_PATTERN_5] = std::move(partition);
  });
}

at::Tensor execute_sdpa_partition(
    const Tensor& key,
    const Tensor& query,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& attn_mask,
    cp_entry& cp) {
  int i = 0;

  cp.inputLLGATensors_[i++].set_data_handle(query.data_ptr());
  cp.inputLLGATensors_[i++].set_data_handle(key.data_ptr());
  if (scale.has_value()) {
    cp.inputLLGATensors_[i++].set_data_handle(scale.value().data_ptr());
  }
  if (attn_mask.has_value()) {
    cp.inputLLGATensors_[i++].set_data_handle(attn_mask.value().data_ptr());
  }
  cp.inputLLGATensors_[i++].set_data_handle(value.data_ptr());

  auto output_tensor = at::empty_like(query);
  cp.outputLLGATensors_[0].set_data_handle(output_tensor.data_ptr());
  cp.cp_.execute(
      onednn_graph::Stream::getStream(),
      cp.inputLLGATensors_,
      cp.outputLLGATensors_);
  return output_tensor;
}

void compile_and_cache_sdpa_pattern_5(
    const partition& partition,
    const Tensor& key,
    const Tensor& query,
    const Tensor& value,
    const Tensor& scale,
    const Tensor& attn_mask,
    cp_entry& cp) {
  cp.inputLogicalTensors_.push_back(
      {0, data_type::f32, query.sizes().vec(), query.strides().vec()});
  cp.inputLogicalTensors_.push_back(
      {1, data_type::f32, key.sizes().vec(), key.strides().vec()});
  cp.inputLogicalTensors_.push_back(
      {4, data_type::f32, scale.sizes().vec(), scale.strides().vec()});
  cp.inputLogicalTensors_.push_back(
      {6, data_type::f32, attn_mask.sizes().vec(), attn_mask.strides().vec()});
  cp.inputLogicalTensors_.push_back(
      {2, data_type::f32, value.sizes().vec(), value.strides().vec()});
  cp.outputLogicalTensors_.push_back(
      {9, data_type::f32, query.sizes().vec(), query.strides().vec()});
  cp.partition_ = partition;
  cp.cp_ = std::move(compile_partition(
      partition, cp.inputLogicalTensors_, cp.outputLogicalTensors_));

  int i = 0;
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      query.data_ptr()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      key.data_ptr()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      scale.data_ptr()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i++],
      onednn_graph::Engine::getEngine(),
      attn_mask.data_ptr()));
  cp.inputLLGATensors_.emplace_back(RunArg(
      cp.inputLogicalTensors_[i],
      onednn_graph::Engine::getEngine(),
      value.data_ptr()));
  i = 0;
  cp.outputLLGATensors_.emplace_back(RunArg(
      cp.outputLogicalTensors_[i],
      onednn_graph::Engine::getEngine(),
      query.data_ptr()));
}

} // end anonymous namespace

Tensor mkldnn_graph_sdpa_pattern(
    const int64_t patternID,
    const Tensor& key,
    const Tensor& query,
    const Tensor& value,
    const c10::optional<Tensor>& scale,
    const c10::optional<Tensor>& attn_mask) {
  // first check cache
  std::vector<int64_t> map_key;
  map_key.reserve(1024);
  map_key.push_back(omp_get_max_threads());
  // Algo ID
  map_key.push_back(patternID);

  map_key.insert(map_key.end(), key.sizes().begin(), key.strides().end());
  map_key.insert(map_key.end(), query.sizes().begin(), query.strides().end());
  map_key.insert(map_key.end(), value.sizes().begin(), value.strides().end());
  if (scale.has_value()) {
    auto scale_val = scale.value();
    map_key.insert(
        map_key.end(), scale_val.sizes().begin(), scale_val.strides().end());
  }
  if (attn_mask.has_value()) {
    auto attn_mask_val = attn_mask.value();
    map_key.insert(
        map_key.end(),
        attn_mask_val.sizes().begin(),
        attn_mask_val.strides().end());
  }

  auto iter = cache_items_map_.find(map_key);
  if (iter == cache_items_map_.end()) {
    cp_entry compiledPartitionEntry;
    auto graph_partition_iter = partition_map_.find(patternID);
    partition graph_partition;
    if (graph_partition_iter == partition_map_.end()) {
      switch (patternID) {
        case ONEDNN_GRAPH_SDPA_PATTERN_5:
          create_graph_sdpa_pattern_5();
      }
      graph_partition_iter = partition_map_.find(patternID);
    }
    graph_partition = graph_partition_iter->second;
    switch (patternID) {
      case ONEDNN_GRAPH_SDPA_PATTERN_5:
        compile_and_cache_sdpa_pattern_5(
            graph_partition,
            key,
            query,
            value,
            scale.value(),
            attn_mask.value(),
            compiledPartitionEntry);
    }
    auto retVal = execute_sdpa_partition(
        key, query, value, scale, attn_mask, compiledPartitionEntry);
    cache_items_list_.push_front(
        key_value_pair_t(map_key, std::move(compiledPartitionEntry)));
    cache_items_map_[map_key] = cache_items_list_.begin();
    if (cache_items_map_.size() > capacity_) {
      auto last = cache_items_list_.end();
      last--;
      cache_items_map_.erase(last->first);
      cache_items_list_.pop_back();
    }
    return retVal;
  } else {
    cache_items_list_.splice(
        cache_items_list_.begin(), cache_items_list_, iter->second);
    cp_entry& cp = iter->second->second;
    return execute_sdpa_partition(key, query, value, scale, attn_mask, cp);
  }
  return query;
}

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_graph_sdpa_pattern"),
      TORCH_FN(mkldnn_graph_sdpa_pattern));
}

} // namespace native
} // namespace at

#endif
