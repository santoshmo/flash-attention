#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

extern __global__ void renorm_kernel(
    __half* out,
    const float2* global_stats,
    const float2* local_stats,
    int B, int H, int len, int D);

torch::Tensor run_unit_renorm(int B, int H, int len, int D) {
  auto opts_f16 = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
  auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  auto out          = torch::ones({B,len,H,D}, opts_f16);          // (b,t,h,d)
  auto global_stats = torch::zeros({B,H,len,2}, opts_f32);         // rowmax=0,sumexp=1
  auto local_stats  = torch::zeros_like(global_stats);

  // set global rowmax=1, sumexp=2 so α should become 0.5*exp(-1)
  global_stats.select(-1,0).fill_(1.f);
  global_stats.select(-1,1).fill_(2.f);

  // local rowmax=0 (already) , sumexp=1 
  local_stats.select(-1,1).fill_(1.f);

  // local rowmax=0, sumexp=1 (already)
  int threads = 256;
  int elems   = B*H*len*D;
  int blocks  = (elems + threads-1)/threads;

  renorm_kernel<<<blocks,threads>>>(
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<float2*>(global_stats.data_ptr<float>()),
      reinterpret_cast<float2*>(local_stats.data_ptr<float>()),
      B,H,len,D);

  return out.cpu();   // copy to host for inspection
}

PYBIND11_MODULE(unit_renorm, m) {
  m.def("run", &run_unit_renorm);
}