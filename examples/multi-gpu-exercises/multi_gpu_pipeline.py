import argparse, math, os, socket, torch, torch.distributed as dist
import torch.multiprocessing as mp
import flash_attn_2_cuda as fa     # ⬅ low-level FlashAttention-2 bindings

# --- utility ---------------------------------------------------------------

def nvlink_sleep(tensor, bandwidth_gb_s: float = 470.0):
    """
    Busy-wait on the current stream for the number of cycles that copying
    `tensor` over NVLink (≈ 470 GB/s on H100) would cost.
    """
    bytes_ = tensor.numel() * tensor.element_size()
    secs = bytes_ / (bandwidth_gb_s * 1e9)
    cycles = int(secs * 1e9)          # 1 cycle ≈ 1 ns on modern GPUs
    if cycles > 0:
        torch.cuda._sleep(cycles)

# --- main worker -----------------------------------------------------------

def worker(rank, world, args):
    sim = args.sim
    dev = 0 if sim else rank
    torch.cuda.set_device(dev)

    # ─────────────────────────── dist init ────────────────────────────────
    backend = "gloo" if sim else "nccl"
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{args.master}:{args.port}",
        rank=rank,
        world_size=world,
    )

    # 2 streams: one for compute, one for “comm”
    s_compute = torch.cuda.Stream()
    s_comm    = torch.cuda.Stream()

    # dummy inputs ---------------------------------------------------------
    B, S, H, D = 1, 2048, 8, 64   # keep it small enough for a single card
    dtype = torch.float16
    q = torch.randn(B, S, H, D, dtype=dtype, device=dev, requires_grad=False)
    k = q.clone(); v = q.clone()
    scale = 1.0 / math.sqrt(D)

    lse_prev = None
    niters = 6

    for it in range(niters):
        # ────────────────── phase A : COMM (prev iter) ───────────────────
        if lse_prev is not None:
            with torch.cuda.stream(s_comm):
                # kick off async all-reduce --> Work handle
                work = dist.all_reduce(lse_prev, async_op=True)
                if sim:                       # CPU Gloo doesn’t touch GPU,
                    nvlink_sleep(lse_prev)    # so inject realistic latency
        else:
            work = None

        # ────────────────── phase B : COMPUTE (this iter) ────────────────
        with torch.cuda.stream(s_compute):
            out, lse, *_ = fa.fwd(
                q, k, v,
                None, None, 0.0, scale,
                False,  # causal
                -1, -1, 0.0, False, None
            )

        # make compute of NEXT iter wait for comm to finish
        if work is not None:
            work.wait()
        s_compute.wait_stream(s_comm)

        lse_prev = lse  # will be reduced in the *next* iteration

    torch.cuda.synchronize()
    if rank == 0:
        print("✓ finished {} iterations (sim={})".format(niters, sim))
    dist.destroy_process_group()

# --- launcher --------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0)); return s.getsockname()[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", action="store_true",
                   help="Run all ranks on cuda:0 with Gloo + fake NVLink")
    p.add_argument("--nproc", type=int, default=8,
                   help="World size (default 8)")
    p.add_argument("--master", default="127.0.0.1")
    p.add_argument("--port", type=int, default=find_free_port())
    args = p.parse_args()

    if args.sim or torch.cuda.device_count() < args.nproc:
        args.sim = True
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("▶ SIMULATION mode:  all ranks share cuda:0")
    else:
        print("▶ REAL multi-GPU mode on", args.nproc, "GPUs")

    mp.spawn(worker, args=(args.nproc, args), nprocs=args.nproc, join=True)

if __name__ == "__main__":
    main()
