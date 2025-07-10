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
    # Allocate a mutable buffer for the NCCL unique ID
    uid_len = getattr(nccl, "NCCL_UNIQUE_ID_BYTES", 128)
    uid = bytearray(uid_len)
    if comm.rank == 0:
        # Root rank fills the buffer with the generated unique ID
        uid_tuple = nccl.get_unique_id()
        # Convert signed ints to bytes (0–255) and copy into the buffer
        uid[:] = bytes((x & 0xFF) for x in uid_tuple)
    # Broadcast the raw bytes to all ranks
    comm.Bcast([uid, MPI.BYTE], root=0)
    return uid

def main():
    # ------------------------------------------------------------------
    # MPI bootstrap (already gives us world-size / rank)
    # ------------------------------------------------------------------
    comm       = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank       = comm.Get_rank()

    # ------------------------------------------------------------------
    # Select a GPU for this rank *before* initializing the NCCL PG
    # ------------------------------------------------------------------
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # ------------------------------------------------------------------
    # Torch process-group (NCCL)
    # ------------------------------------------------------------------
    setup_torch_pg(world_size, rank)

    # fix RNG for reproducibility
    torch.manual_seed(1234 + rank)

    # ------------------------------------------------------------------
    # Build *local* Q/K/V tensors: each rank owns only its slice.
    # Placeholder: we simply assign one contiguous chunk per rank.
    # TODO: Replace with a proper load-balancing / sharding strategy.
    # ------------------------------------------------------------------
    B, S, H, D = 1, pow(2, 22), 8, 64              # batch, heads, dim per head
    print(f"Sequence length {S}")
    tokens_per_rank = S // world_size            # sequence length handled by EACH rank
    chunk_size      = tokens_per_rank // 2       # two chunk locally for now (assume it'll be a power of four)

    q = torch.randn(B, tokens_per_rank, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    uid = get_shared_nccl_uid()

    out = pipelined_fa.pipelined_fa(
        q, k, v,
        chunk_size,
        world_size,
        rank,
        bytes(uid)
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