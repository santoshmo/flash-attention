import torch

# create a dummy activation
act = torch.arange(0, 8, device="cuda", dtype=torch.float32)

# two streams for “GPU 0” and “GPU 1”
stream0 = torch.cuda.Stream()
stream1 = torch.cuda.Stream()

# split activation in half
act0 = act[:4]
act1 = act[4:]

# allocate placeholders for the “received” halves
recv0 = torch.empty_like(act0)
recv1 = torch.empty_like(act1)

# compute on slice 0 and copy to recv1 (simulating send to GPU 1)
with torch.cuda.stream(stream0):
    out0 = act0 * 3.0  # pretend this is a model on GPU 0
    torch.cuda.current_stream().record_event()
    # async “send” to slice 1
    torch.cuda.current_stream().wait_stream(stream1)
    recv1.copy_(out0, non_blocking=True)

# compute on slice 1 and copy to recv0
with torch.cuda.stream(stream1):
    out1 = act1 + 5.0  # pretend this is a model on GPU 1
    torch.cuda.current_stream().record_event()
    torch.cuda.current_stream().wait_stream(stream0)
    recv0.copy_(out1, non_blocking=True)

# synchronize both to ensure compute & copy done
torch.cuda.synchronize()

print("recv0 (from slice 1):", recv0.tolist())
print("recv1 (from slice 0):", recv1.tolist())
