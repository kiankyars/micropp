import torch
from comms import init_distributed, PipelineComms

def run_ping_pong():
    rank, world_size, device = init_distributed()
    print(device)
    comms = PipelineComms(rank, world_size)

    # Simple 2-GPU Logic
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        print(f"Rank 0: Sending {tensor}")
        comms.send_forward(tensor)
    elif rank == 1:
        # Must know shape in advance!
        received = comms.recv_forward(shape=(3,))
        print(f"Rank 1: Received {received}")

if __name__ == "__main__":
    run_ping_pong()