# test_pipelined_fa.py
import os
import torch
import torch.distributed as dist
from mpi4py import MPI                  
import cupy.cuda.nccl as nccl
import pipelined_fa

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def setup_torch_pg(world_size, rank):
    """
    Initialise a (CPU-only) Gloo process group that piggy-backs on MPI
    environment variables.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"       # same node
    os.environ["MASTER_PORT"] = "29500"           # free port
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        init_method="env://"
    )

def get_shared_nccl_uid():
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        uid = nccl.get_unique_id()
    else:
        uid = bytearray(128)       # placeholder
    comm.Bcast(uid, root=0)
    return uid

def main():
    # ------------------------------------------------------------------
    # MPI bootstrap (already gives us world-size / rank)
    # ------------------------------------------------------------------
    comm       = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank       = comm.Get_rank()

    # ------------------------------------------------------------------
    # Torch process-group (Gloo) – CPU only, so 1 GPU is fine
    # ------------------------------------------------------------------
    setup_torch_pg(world_size, rank)

    # fix GPU and RNG for reproducibility
    torch.cuda.set_device(0)
    torch.manual_seed(1234 + rank)

    # ------------------------------------------------------------------
    # Build dummy Q/K/V tensors and run the extension
    # ------------------------------------------------------------------
    B, S, H, D  = 1, 128, 8, 64        # (batch, tokens, heads, dim)
    chunk_size  = int(S / world_size) # should equal 64 now

    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    uid = get_shared_nccl_uid()
    comm.Bcast(uid, root=0)

    out = pipelined_fa.pipelined_fa(
        q, k, v,
        chunk_size,
        1,
        0
    )

    # ref = flash_attn_interface.flash_attn_func(q, k, v)
    # torch.allclose(out, ref, atol=1e-3)
    # ------------------------------------------------------------------
    # Verify something simple and print a checksum
    # ------------------------------------------------------------------
    checksum = out.float().norm().item()
    print(f"[Rank {rank}]  output L2-norm = {checksum:.6f}")

    # optional barrier so the script only finishes when all ranks did
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()