import torch
import torch.optim as optim
import time

# Import our modules
from comms import init_distributed, PipelineComms
from model import ShardedMLP
from profiled_schedule import naive_pipeline_step, gpipe_pipeline_step, onef_oneb_pipeline_step
from profiler import PipelineProfiler

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
TOTAL_LAYERS = 16
STEPS = 50
CHUNKS = 4

# 1. Setup Distributed Environment
rank, world_size, device = init_distributed()
comms = PipelineComms(rank, world_size)
profiler = PipelineProfiler(rank)

torch.manual_seed(42)
# Each rank needs to "skip" the random numbers used by previous ranks
for i in range(rank * (TOTAL_LAYERS // world_size) * 2):  # 2 params per layer (weight, bias)
    torch.randn(1)  # Consume RNG state

if rank == 0:
    print(f"--- Starting Micro PP on {world_size} Processes ({device}) ---")

# 2. Initialize the Sharded Model
model = ShardedMLP(HIDDEN_DIM, TOTAL_LAYERS, rank, world_size).to(device)

# 3. Setup Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 4. Only Rank 0 loads the data.
if rank == 0:
    fixed_input = torch.randn((BATCH_SIZE, HIDDEN_DIM)).to(device)
else:
    fixed_input = BATCH_SIZE
# 5. Only the Last Rank needs the targets to calc loss.
if rank == world_size - 1:
    # We want the model to learn to classify these random vectors into class '0' or '1'
    fixed_target = torch.randint(0, 2, (BATCH_SIZE,)).to(device)
else:
    fixed_target = None

start_time = time.time()
model.train()
for step in range(STEPS):
    optimizer.zero_grad()
    if rank == world_size - 1:
        loss = onef_oneb_pipeline_step(model, comms, profiler, fixed_input, fixed_target, HIDDEN_DIM, CHUNKS, device)
    else:
        onef_oneb_pipeline_step(model, comms, profiler, fixed_input, fixed_target, HIDDEN_DIM, CHUNKS, device)
    optimizer.step()
    if rank == world_size - 1 and step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss.item():.6f}")
# Clean up
if rank == world_size - 1:
    print("--- Training Complete ---")
    duration = time.time() - start_time
    print(f"Final Loss: {loss.item():.6f} | Time: {duration:.3f}s")
profiler.print_summary()
torch.distributed.destroy_process_group()