import torch
from comms import init_distributed, PipelineComms

def run_ping_pong():
    rank, world_size, device = init_distributed()
    print(rank, world_size, device)
    comms = PipelineComms(rank, world_size)

    # Simple 2-GPU Logic
    if rank == 0:
        tensor = torch.rand(3).to(device)
        print(f"Rank 0: Sending {tensor}")
        comms.send_forward(tensor)
    elif rank == 1:
        # Must know shape in advance!
        shape = (3,)
        received = comms.recv_forward(shape, device)
        print(f"Rank 1: Received {received}")

if __name__ == "__main__":
    run_ping_pong()