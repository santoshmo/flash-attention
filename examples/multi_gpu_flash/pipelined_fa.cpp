/*
 * flash_pipelined.cu
 * Implements pipelined Tree Attention via FlashAttention2 with overlapped NCCL collectives.
 * Strategy: per-chunk FlashAttention kernels write rowmax/sumexp; separate comm stream
 * launches ncclReduceScatter+ncclAllReduce on float2 stats buffers; compute and comm
 * overlap via CUDA streams and events.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cmath>
#include <cuda_fp16.h>
#include <pybind11/pybind11.h>

#include "flash.h"    // flash_attn fprop kernel declarations
#include "flash_api.h"

#include <cstdio>

#define CUDA_CHECK(msg)                                                \
  {                                                                    \
    cudaError_t _e = cudaGetLastError();                               \
    if (_e != cudaSuccess) {                                           \
      printf("CUDA-ERROR %s : %s (%s:%d)\n", msg,                      \
             cudaGetErrorString(_e), __FILE__, __LINE__);              \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  }

#define NCCL_CHECK(cmd)                                                \
  {                                                                    \
    ncclResult_t _r = cmd;                                             \
    if (_r != ncclSuccess) {                                           \
      printf("NCCL-ERROR %s : %s (%s:%d)\n", #cmd,                      \
             ncclGetErrorString(_r), __FILE__, __LINE__);              \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  }

// Forward declarations of CUDA kernels implemented in renorm_kernel.cu
__global__ void renorm_kernel_v2(
  __half* out,                   // output slice (fp16)
  const float* global_rowmax,    // rowmax AFTER NCCL
  const float* global_sumexp,    // sumexp AFTER NCCL
  const float* local_rowmax,     // per‐rank rowmax BEFORE NCCL
  const float* local_sumexp,     // per‐rank sumexp BEFORE NCCL
  int B,
  int H,
  int len,
  int D);

__global__ void renorm_kernel(
  __half* out,            // output slice (fp16)
  const float2* global_stats, // rowmax/sumexp AFTER NCCL
  const float2* local_stats,  // per‐rank values BEFORE NCCL
  int B,
  int H,
  int len,
  int D);

//----------------------------------------------------------------------
// Host-side wrapper for pipelined FlashAttention + NCCL
//----------------------------------------------------------------------

struct PipelinedFA2 {
  ncclComm_t comm;
  int world_size;
  int rank;
  cudaStream_t compute_stream;
  cudaStream_t comm_stream;
  cudaEvent_t  events[2];
  // double-buffered stats: [2][B][H][chunk][2 scalars]
  at::Tensor lse_buf;
  at::Tensor rowmax_buf;
  at::Tensor sumexp_buf;
  // local copy of stats for renorm: same shape as stats_buf
  at::Tensor local_lse_buf;
  at::Tensor local_rowmax_buf;
  at::Tensor local_sumexp_buf;

  PipelinedFA2(ncclComm_t comm_, int world_size_, int rank_)
    : comm(comm_), world_size(world_size_), rank(rank_) {
    // create streams
    cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&comm_stream,    cudaStreamNonBlocking);
    for (int i = 0; i < 2; ++i) {
      cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    }
  }

  ~PipelinedFA2() {
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(comm_stream);
    for (int i = 0; i < 2; ++i) cudaEventDestroy(events[i]);
  }

  // args: q,k,v,out are physically contiguous [B, S, H, D]
  // rowmax and sumexp returned in rowmax/sumexp tensors
  void run(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &rowmax,
    at::Tensor &sumexp,
    int chunk_size
  ) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Inputs must be CUDA tensors");
    
    auto B = q.size(0), S = q.size(1), H = q.size(2), D = q.size(3);
    int n_chunks = ceil(S / chunk_size);

    // prepare double-buffered stats buffer on device (rowmax,sumexp)
    // Flash Attention expects [B, H, seqlen] format for stats
    auto opts = q.options().dtype(torch::kFloat32);
    lse_buf   = torch::empty({2, B, H, chunk_size}, opts);
    rowmax_buf = torch::empty_like(lse_buf);
    sumexp_buf = torch::empty_like(rowmax_buf);

    // keep local copies before NCCL
    local_lse_buf = torch::empty_like(lse_buf);
    local_rowmax_buf = torch::empty_like(rowmax_buf);
    local_sumexp_buf = torch::empty_like(sumexp_buf);

    for (int chunk = 0; chunk < n_chunks + 1; ++chunk) {
      int buf_idx = chunk % 2; // TODO(santoshmo): needs to change? 

      // 1) launch FlashAttn kernel for chunk on compute_stream
      if (chunk < n_chunks) {
        int start = chunk * chunk_size;
        int len   = std::min(chunk_size, int(S - start));
        auto q_slice   = q.narrow(1, start, len).contiguous(); // (1, 0, 64)
        auto k_slice   = k.narrow(1, start, len).contiguous(); // (1, 0, 64)
        auto v_slice   = v.narrow(1, start, len).contiguous(); // (1, 0, 64)
        auto out_slice = out.narrow(1, start, len).contiguous(); // (1, 0, 64)
        
        // Create slice views for stats buffers with correct layout [B, H, len]
        auto lse_fa_slice = lse_buf[buf_idx].narrow(2, 0, len);
        auto rowmax_fa_slice = rowmax_buf[buf_idx].narrow(2, 0, len);
        auto sumexp_fa_slice = sumexp_buf[buf_idx].narrow(2, 0, len);
        
        flash::Flash_fwd_params params; 
        flash::set_params_fprop(params,
                         B, len, len,
                         len, len,
                         H, H, D, D,
                         q_slice, k_slice, v_slice, out_slice,
                         nullptr, nullptr, nullptr,
                         nullptr,
                         lse_fa_slice.data_ptr<float>(),
                         rowmax_fa_slice.data_ptr<float>(),
                         sumexp_fa_slice.data_ptr<float>(),
                         0.f,
                         1.0f/ sqrtf((float)D),
                         -1, -1, 0.0f);

        run_mha_fwd(params, compute_stream);
        CUDA_CHECK("run_mha_fwd");
        
        // copy back the contiguous output slice if needed
        if (!out_slice.is_same(out_slice)) {
          out_slice.copy_(out_slice);
        }

        // copy stats to local buffer before NCCL modifies them
        // Reuse the same slices from Flash Attention call above
        local_rowmax_buf[buf_idx].narrow(2, 0, len).copy_(rowmax_fa_slice);
        local_sumexp_buf[buf_idx].narrow(2, 0, len).copy_(sumexp_fa_slice);

        cudaEventRecord(events[buf_idx], compute_stream);
      }

      // 2) launch NCCL on previous chunk's stats
      if (chunk > 0) {
        int prev = (chunk - 1) % 2;
        int start_prev = (chunk - 1) * chunk_size;
        int len_prev   = std::min(chunk_size, int(S - start_prev));

        // COMMENTED OUT: wait for NCCL stream - for renorm kernel testing only
        cudaStreamWaitEvent(comm_stream, events[prev], 0);

        // Get slices for the previous chunk with correct dimensions
        auto g_rowmax_slice = rowmax_buf[prev].narrow(2, 0, len_prev);
        auto g_sumexp_slice = sumexp_buf[prev].narrow(2, 0, len_prev);
        auto l_rowmax_slice = local_rowmax_buf[prev].narrow(2, 0, len_prev);
        auto l_sumexp_slice = local_sumexp_buf[prev].narrow(2, 0, len_prev);
        
        float* g_rowmax = g_rowmax_slice.data_ptr<float>();
        float* g_sumexp = g_sumexp_slice.data_ptr<float>();
        float* l_rowmax = l_rowmax_slice.data_ptr<float>();
        float* l_sumexp = l_sumexp_slice.data_ptr<float>();

        size_t count = static_cast<size_t>(B) * H * len_prev;

        if (comm != nullptr && world_size > 1) {
          NCCL_CHECK(ncclGroupStart());
          NCCL_CHECK(ncclAllReduce(g_rowmax, g_rowmax, count, ncclFloat32, ncclMax, comm, comm_stream));
          NCCL_CHECK(ncclAllReduce(g_sumexp, g_sumexp, count, ncclFloat32, ncclSum, comm, comm_stream));
          NCCL_CHECK(ncclGroupEnd());
          
          // record completion once NCCL ops finish
          cudaEventRecord(events[prev], comm_stream);
        } else {
          // Single-rank or mock run: global stats == local stats
          g_rowmax_slice.copy_(l_rowmax_slice);
          g_sumexp_slice.copy_(l_sumexp_slice);

          // Immediately mark event as complete from compute stream
          cudaEventRecord(events[prev], compute_stream);
        }
        
        // 3) enqueue local renorm on compute_stream
        cudaStreamWaitEvent(compute_stream, events[prev], 0);
        
        // Create combined stats buffers like the working unit test
        // We need to interleave rowmax/sumexp as float2 pairs
        auto global_stats_combined = torch::empty({B, H, len_prev, 2}, rowmax_buf[0].options());
        auto local_stats_combined = torch::empty_like(global_stats_combined);
        
        // Copy data: select(-1,0) gets rowmax, select(-1,1) gets sumexp
        global_stats_combined.select(-1, 0).copy_(g_rowmax_slice.view({B, H, len_prev}));
        global_stats_combined.select(-1, 1).copy_(g_sumexp_slice.view({B, H, len_prev}));
        local_stats_combined.select(-1, 0).copy_(l_rowmax_slice.view({B, H, len_prev}));
        local_stats_combined.select(-1, 1).copy_(l_sumexp_slice.view({B, H, len_prev}));
        
        int threads = 256;
        int elems = B * len_prev * H * D;  // Fix: len_prev comes before H  
        int blocks = (elems + threads - 1) / threads;
        
        // Get the output slice for the previous chunk and make it contiguous
        auto out_slice_prev = out.narrow(1, start_prev, len_prev).contiguous();
        __half* out_ptr = reinterpret_cast<__half*>(out_slice_prev.data_ptr<at::Half>());
        
        renorm_kernel<<<blocks, threads, 0, compute_stream>>>(
          out_ptr, 
          reinterpret_cast<float2*>(global_stats_combined.data_ptr<float>()),
          reinterpret_cast<float2*>(local_stats_combined.data_ptr<float>()),
          B, H, len_prev, D
        );
        CUDA_CHECK("renorm_kernel");
        
        auto orig_out_slice = out.narrow(1, start_prev, len_prev);
        if (!orig_out_slice.is_same(out_slice_prev)) {
          orig_out_slice.copy_(out_slice_prev);
        }
      }
    }

    cudaStreamSynchronize(compute_stream);
    cudaStreamSynchronize(comm_stream);
  }
};

// Python binding
torch::Tensor pipelined_fa(
  torch::Tensor q,
  torch::Tensor k,
  torch::Tensor v,
  int chunk_size,
  int nccl_world_size,
  int nccl_rank,
  pybind11::bytes uid_bytes = pybind11::bytes()
) {
  ncclUniqueId uid;
  if (nccl_world_size == 1) {
    // Single-rank: generate a throwaway UID
    NCCL_CHECK(ncclGetUniqueId(&uid));
  } else {
    std::string uid_str = uid_bytes;  // bytes → std::string (keeps raw data)
    if (uid_str.empty()) {
      // Back-compat path (old 6-arg call): each rank would generate its own UID → error.
      // We throw to surface the issue clearly.
      throw std::runtime_error("A shared NCCL unique ID must be provided when nccl_world_size > 1");
    }
    if (uid_str.size() != NCCL_UNIQUE_ID_BYTES) {
      throw std::runtime_error("uid_bytes must have NCCL_UNIQUE_ID_BYTES length");
    }
    memcpy(&uid, uid_str.data(), NCCL_UNIQUE_ID_BYTES);
  }

  ncclComm_t comm = nullptr;
  NCCL_CHECK(ncclCommInitRank(&comm, nccl_world_size, uid, nccl_rank));
  
  auto out = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));
  auto rowmax = torch::empty_like(lse);
  auto sumexp = torch::empty_like(rowmax);

  PipelinedFA2 engine(comm, nccl_world_size, nccl_rank);
  engine.run(q, k, v, out, lse, rowmax, sumexp, chunk_size);

  return out;
}

PYBIND11_MODULE(pipelined_fa, m) {
  namespace py = pybind11;
  m.def("pipelined_fa", &pipelined_fa, 
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("chunk_size"),
        py::arg("nccl_world_size"), py::arg("nccl_rank"),
        py::arg("uid_bytes") = py::bytes(),
        "Pipelined FlashAttention2 with overlapped NCCL");
}
