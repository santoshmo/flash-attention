#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// Define the renorm kernel with separate arrays
__global__ void renorm_kernel_v2(
    __half* out,                   // [B, len, H, D] - contiguous slice
    const float* global_rowmax,    // [B * H * len] after NCCL
    const float* global_sumexp,    // [B * H * len] after NCCL
    const float* local_rowmax,     // same shape, before NCCL (per-rank)
    const float* local_sumexp,     // same shape, before NCCL (per-rank)
    int B, int H, int len, int D) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * len * H * D;  // Fixed: len comes before H in tensor layout
  
  // Debug first few threads
  if (idx < 10) {
    printf("KERNEL DEBUG [%d]: B=%d, H=%d, len=%d, D=%d, total=%d\n", 
           idx, B, H, len, D, total);
    printf("KERNEL DEBUG [%d]: pointers - out=%p, g_rowmax=%p, g_sumexp=%p, l_rowmax=%p, l_sumexp=%p\n", 
           idx, out, global_rowmax, global_sumexp, local_rowmax, local_sumexp);
  }
  
  if (idx >= total) {
    if (idx < total + 10) {
      printf("KERNEL DEBUG [%d]: thread %d >= total %d, returning\n", idx, idx, total);
    }
    return;
  }

  // Decompose flattened index to (b, t, h, d) - matching [B, len, H, D] layout
  int d = idx % D;
  int tmp = idx / D;
  int h = tmp % H;
  tmp = tmp / H;
  int t = tmp % len;
  int b = tmp / len;

  // Bounds checking
  if (b >= B || t >= len || h >= H || d >= D) {
    if (idx < 10) {
      printf("KERNEL DEBUG [%d]: bounds check failed: b=%d/%d, t=%d/%d, h=%d/%d, d=%d/%d\n", 
             idx, b, B, t, len, h, H, d, D);
    }
    return;
  }

  // Stats index: stats are stored as [B, H, len] -> (b * H + h) * len + t
  int sidx = (b * H + h) * len + t;
  int stats_total = B * H * len;
  
  if (idx < 10) {
    printf("KERNEL DEBUG [%d]: computed sidx=%d, stats_total=%d (b=%d, h=%d, t=%d)\n", 
           idx, sidx, stats_total, b, h, t);
  }
  
  // Add bounds check for stats arrays
  if (sidx >= stats_total || sidx < 0) {
    printf("KERNEL ERROR [%d]: sidx=%d out of bounds [0, %d) (b=%d, h=%d, t=%d, len=%d)\n", 
           idx, sidx, stats_total, b, h, t, len);
    return;
  }

  // Check if we can safely read from stats arrays
  float rowmax_global, sumexp_global, rowmax_local, sumexp_local;
  
  // Add explicit memory access with error checking
  rowmax_global = global_rowmax[sidx];
  sumexp_global = global_sumexp[sidx];
  rowmax_local = local_rowmax[sidx];
  sumexp_local = local_sumexp[sidx];
  
  if (idx < 5) {
    printf("KERNEL DEBUG [%d]: stats values - g_max=%.3f, g_sum=%.3f, l_max=%.3f, l_sum=%.3f\n", 
           idx, rowmax_global, sumexp_global, rowmax_local, sumexp_local);
  }

  // Avoid division by zero and invalid exponentials
  if (sumexp_global <= 0.0f || sumexp_local <= 0.0f) {
    if (idx < 5) {
      printf("KERNEL DEBUG [%d]: invalid sumexp values, skipping\n", idx);
    }
    return;
  }

  // Compute scaling factor
  float alpha = __expf(rowmax_local - rowmax_global) * (sumexp_local / sumexp_global);
  
  if (idx < 5) {
    printf("KERNEL DEBUG [%d]: alpha=%.6f, accessing out[%d]\n", idx, alpha, idx);
  }

  // Check if output access is safe
  if (out == nullptr) {
    printf("KERNEL ERROR [%d]: out pointer is null\n", idx);
    return;
  }

  // Scale output element
  float out_f = __half2float(out[idx]);
  out_f *= alpha;
  out[idx] = __float2half(out_f);
  
  if (idx < 5) {
    printf("KERNEL DEBUG [%d]: completed successfully\n", idx);
  }
}

// Keep original kernel for backward compatibility
__global__ void renorm_kernel(
    __half* out,                 // [B, len, H, D] - contiguous slice
    const float2* global_stats,  // [B * H * len] rowmax/sumexp after NCCL
    const float2* local_stats,   // same shape, before NCCL (per-rank)
    int B, int H, int len, int D) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = B * len * H * D;  // Fixed: len comes before H in tensor layout
  if (idx >= total) return;

  // Decompose flattened index to (b, t, h, d) - matching [B, len, H, D] layout
  int d = idx % D;
  int tmp = idx / D;
  int h = tmp % H;
  tmp = tmp / H;
  int t = tmp % len;
  int b = tmp / len;

  // Bounds checking
  if (b >= B || t >= len || h >= H || d >= D) return;

  // Stats index: stats are stored as [B, H, len] -> (b * H + h) * len + t
  int sidx = (b * H + h) * len + t;

  float2 g = global_stats[sidx];
  float2 l = local_stats[sidx];

  float rowmax_global = g.x;
  float sumexp_global = g.y;
  float rowmax_local = l.x;
  float sumexp_local = l.y;

  // Avoid division by zero and invalid exponentials
  if (sumexp_global <= 0.0f || sumexp_local <= 0.0f) return;

  // Compute scaling factor
  float alpha = __expf(rowmax_local - rowmax_global) * (sumexp_local / sumexp_global);

  // Scale output element
  float out_f = __half2float(out[idx]);
  out_f *= alpha;
  out[idx] = __float2half(out_f);
}
