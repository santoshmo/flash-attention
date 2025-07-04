// csrc/flash_attn/include/flash_api.h
#pragma once
#include <ATen/ATen.h>
// #include <torch/python.h>
// #include <torch/nn/functional.h>

struct Flash_fwd_params;  // forward-declare

namespace FLASH_NAMESPACE {
  // exactly match this to the definition in flash_api.cpp
  void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      void *softmax_rowmax_d,
                      void *softmax_sumexp_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false);
  void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false);
}