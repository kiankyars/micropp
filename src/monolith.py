import time
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50

# 2. Manual Split (Mental Model for PP)
class Part1(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth/2):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x, targets):
        return self.net(x)

class Part2(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth/2):
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

part1 = Part1(HIDDEN_DIM, TOTAL_LAYERS)
part2 = Part2(HIDDEN_DIM, TOTAL_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate one fixed batch to overfit
fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM)
# low is inclusive, high is exclusive
fixed_target = torch.randint(0, 2, (BATCH_SIZE,))

# 4. Training Loop
print("--- Training Monolith (Ground Truth) ---")
start_time = time.time()
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    # Simple forward and backward
    loss = model(fixed_input, fixed_target)
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"Step {step} | Loss: {loss:.4f}")
    # The Training Step
    # 1. Forward
    hidden = part1(fixed_input) 
    # TEACHING MOMENT: This 'hidden' is what will be sent via dist.send
    logits = part2(hidden)
    loss = criterion(logits, fixed_target)

    # 2. Backward
    loss.backward()
    # TEACHING MOMENT: hidden.grad is what will be sent via dist.send_backward

duration = time.time() - start_time
print(f"Final Loss: {loss.item():.6f} Time: {duration:.3f}s")