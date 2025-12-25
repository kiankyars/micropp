import torch.nn as nn

class ShardedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, total_layers, rank, world_size):
        super().__init__()
        
        # 1. Calculate how many layers THIS GPU is responsible for
        layers_per_gpu = total_layers // world_size
        
        self.rank = rank
        self.is_first = (rank == 0)
        self.is_last = (rank + 1 == world_size)
        
        # 2. Build the local stack of layers
        layers = []
        for _ in range(layers_per_gpu):
            # For a simple MLP, every layer looks the same
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets=None):
        # Run the local chunk of the network
        x = self.net(x)
        
        # Only the last GPU calculates loss
        if self.is_last and targets is not None:
            return self.loss_fn(x, targets)
        
        # Everyone else just returns the hidden states (activations)
        return x