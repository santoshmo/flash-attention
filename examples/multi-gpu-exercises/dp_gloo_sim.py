import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_rank(rank, world_size):
    # bind every process to cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )

    # simple input per rank
    x = torch.ones(5, device="cuda") * (rank + 1)  
    # “model” is just a scalar multiply for demo
    y = x * 2.0  
    # all-reduce sum across all ranks
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    # divide to get the mean
    y /= world_size

    print(f"[rank {rank}] y = {y.tolist()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4  # pretend we have 4 GPUs
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)
