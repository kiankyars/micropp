import time
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# 2. The Monolithic Model
# This is what we will eventually "shard" across multiple GPUs
class MonolithicMLP(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        logits = self.net(x)
        return self.loss_fn(logits, targets)

# 3. Setup
torch.manual_seed(42)

model = MonolithicMLP(HIDDEN_DIM, TOTAL_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate one fixed batch to overfit
fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
# low is inclusive, high is exclusive
fixed_target = torch.randint(0, 2, (BATCH_SIZE,))

# 4. Training Loop
print("--- Training Monolith (Ground Truth) ---")
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    start_time = time.time()
    # Simple forward and backward
    loss = model(fixed_input, fixed_target)
    loss.backward()
    optimizer.step()
    duration = time.time() - start_time
    
    if step % 5 == 0:
        print(f"Step {step} | Loss: {loss:.4f} | Time: {duration:.3f}s")

print(f"Final Monolith Loss: {loss.item():.6f}")