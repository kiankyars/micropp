import torch
import torch.distributed as dist
import os

def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # 1. Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # 2. Set Device
    if torch.cuda.is_available():
        # each conditional statement returns the device type
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        # >>> torch.device("mps")
        # device(type='mps')
        device = torch.device("mps")
    elif torch.cpu.is_available():
        device = torch.device("cpu")
    else:
        exit()
    
    # 3. Initialize Group
    if torch.cuda.is_available():
        # Crucial: Pin this process to a specific GPU
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:        
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    
    return rank, world_size, device

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

    def recv_forward(self, shape, dtype, device):
        """Receive activation from the previous GPU."""
        if self.prev_rank is None:
            return None # Rank 0 generates its own data
        
        # We must allocate an empty buffer to receive the data
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.prev_rank)
        return tensor

    def send_backward(self, tensor):
        """Send gradients back to the previous GPU."""
        if self.prev_rank is not None:
            dist.send(tensor.contiguous(), dst=self.prev_rank)

    def recv_backward(self, shape, dtype, device):
        """Receive gradients from the next GPU."""
        if self.next_rank is None:
            return None # Last Rank generates gradients from Loss
        
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.next_rank)
        return tensor