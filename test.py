import torch
device = torch.device("mps", 0)
torch.distributed.init_process_group(device_id=device, rank=0)