import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import flash_attn_2_cuda as fa

def run_rank(rank, world_size):
    # Force every rank onto the same physical GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )

    dev = torch.device("cuda:0")
    dtype = torch.float16
    B, S, H, D = 1, 2048, 8, 64
    total_len = S
    # carve the sequence into world_size chunks
    chunk_size = total_len // world_size
    num_chunks = (total_len + chunk_size - 1) // chunk_size

    # next/prev ranks in the ring
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # two streams per rank
    compute_stream = torch.cuda.Stream(device=dev)
    comm_stream    = torch.cuda.Stream(device=dev)
    # we’ll reuse one event per chunk
    compute_event = torch.cuda.Event()

    # input tensors
    q = torch.randn(B, S, H, D, dtype=dtype, device=dev)
    k = q.clone(); v = q.clone()
    scale = 1.0 / math.sqrt(D)

    # storage for outstanding requests if you need to sync later
    send_reqs = []
    recv_reqs = []
    recv_buffers = []

    for c in range(num_chunks):
        start = c * chunk_size
        end   = min(start + chunk_size, total_len)

        # 1) Compute chunk c on compute_stream
        with torch.cuda.stream(compute_stream):
            q_c = q[:, start:end, :, :]
            k_c = k[:, start:end, :, :]
            v_c = v[:, start:end, :, :]
            out_c, lse_c, rowmax_c, sumexp_c, p_c, rng_c = fa.fwd(
                q_c, k_c, v_c,
                None, None, 0.0, scale,
                False,  # not causal
                -1, -1, 0.0, False, None
            )
            # mark completion of this chunk’s compute
            compute_event.record(compute_stream)

        # 2) On comm_stream, wait and then send/recv
        with torch.cuda.stream(comm_stream):
            # wait until compute done
            comm_stream.wait_event(compute_event)

            # non-blocking send of our out_c to next_rank
            send_reqs.append(dist.isend(out_c, dst=next_rank))

            # post non-blocking recv from prev_rank
            recv_buf = torch.empty_like(out_c)
            recv_buffers.append(recv_buf)
            recv_reqs.append(dist.irecv(recv_buf, src=prev_rank))

        # Next iteration of the loop will start compute on chunk c+1
        # while chunk c is in-flight on comm_stream.

    # 3) wait for all outstanding comms to finish
    for req in send_reqs + recv_reqs:
        req.wait()

    # Optionally: do something with recv_buffers here
    # e.g. print the first chunk you got back
    if recv_buffers:
        print(f"Rank {rank} received chunk[0] sum:", recv_buffers[0].sum().item())

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_rank, args=(world_size,), nprocs=world_size, join=True)