import torch
import torch.distributed as dist
import os

def init_distributed(rank, world_size, local_rank):
    """
    Standard PyTorch boilerplate to set up the process group.
    These variables are usually set by torchrun or slurm
    """
    # Map the process to a specific GPU
    torch.cuda.set_device(local_rank)
    
    # Initialize the group (NCCL is the backend for NVIDIA GPUs)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    return rank, world_size

class PipelineComms:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # Define Neighbors
        # If I am Rank 0, I have no previous neighbor (None)
        self.prev_rank = rank - 1 if rank > 0 else None
        # If I am the last Rank, I have no next neighbor (None)
        self.next_rank = rank + 1 if rank < world_size - 1 else None

    def send_forward(self, tensor):
        """Send activation to the next GPU."""
        if self.next_rank is not None:
            # .contiguous() is required before sending
            dist.send(tensor.contiguous(), dst=self.next_rank)

    def recv_forward(self, shape, dtype=torch.float32):
        """Receive activation from the previous GPU."""
        if self.prev_rank is None:
            return None # Rank 0 generates its own data
        
        # We must allocate an empty buffer to receive the data
        tensor = torch.zeros(shape, dtype=dtype, device='cuda')
        dist.recv(tensor, src=self.prev_rank)
        return tensor

    def send_backward(self, tensor):
        """Send gradients back to the previous GPU."""
        if self.prev_rank is not None:
            dist.send(tensor.contiguous(), dst=self.prev_rank)

    def recv_backward(self, shape, dtype=torch.float32):
        """Receive gradients from the next GPU."""
        if self.next_rank is None:
            return None # Last Rank generates gradients from Loss
        
        tensor = torch.zeros(shape, dtype=dtype, device='cuda')
        dist.recv(tensor, src=self.next_rank)
        return tensor