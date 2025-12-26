'''Imagine main.py is a sheet of music.

torchrun is the conductor.

The Processes are 4 different musicians.

The conductor hands the same sheet of music to all 4 musicians.

local_rank is the instruction at the top of the sheet:

Musician 1's sheet says: "You are playing the Violin (GPU 0)."

Musician 2's sheet says: "You are playing the Viola (GPU 1)."

They all read the same notes (code), but they play on different instruments (GPUs) because of that initial setup instruction.'''
import torch
import torch.optim as optim
import time

# Import your modules
from comms import init_distributed, PipelineComms
from model import ShardedMLP
from schedule import naive_pipeline_step

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
LAYERS_TOTAL = 16
STEPS = 10

def main():
    # 1. Setup Distributed Environment
    rank, world_size, device = init_distributed()
    comms = PipelineComms(rank, world_size)

    if rank == 0:
        print(f"--- Starting Micro PP on {world_size} Processes (Mac/CPU) ---")

    # 2. Initialize the Sharded Model
    # Each process only initializes its specific slice of layers
    model = ShardedMLP(
        input_dim=HIDDEN_DIM, 
        hidden_dim=HIDDEN_DIM, 
        total_layers=LAYERS_TOTAL, 
        rank=rank, 
        world_size=world_size
    ).to(device)

    # 3. Setup Optimizer
    # We only optimize the parameters present on THIS device
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 4. Training Loop
    model.train()
    for step in range(STEPS):
        optimizer.zero_grad()
        
        # --- Data Generation ---
        # In Pipeline Parallelism, usually only Rank 0 loads the data.
        # Everyone else passes 'None' into the step function.
        if rank == 0:
            # Create dummy input: (Batch, Hidden)
            data = torch.randn(BATCH_SIZE, HIDDEN_DIM).to(device)
        else:
            data = None
            
        # Target (Labels)
        # Usually only the Last Rank needs the targets to calc loss.
        if rank == world_size - 1:
            # Dummy targets: integers [0, HIDDEN_DIM)
            targets = torch.randint(0, HIDDEN_DIM, (BATCH_SIZE,)).to(device)
        else:
            targets = None

        # --- The Pipeline Step ---
        start_time = time.time()
        
        # This function handles the Send/Recv/Compute orchestration
        loss = naive_pipeline_step(model, comms, data, targets, HIDDEN_DIM, device)
        
        # Optimizer Step (All ranks do this locally after backward pass completes)
        optimizer.step()
        
        duration = time.time() - start_time

        # --- Logging ---
        # Only the last rank (who calculates loss) can print the loss value
        if rank == world_size - 1:
            print(f"[Step {step+1}/{STEPS}] Loss: {loss:.4f} | Time: {duration:.3f}s")
        elif rank == 0:
            # Rank 0 just lets us know it's alive
            print(f"[Step {step+1}/{STEPS}] Rank 0 Finished")

    # Clean up
    if rank == 0:
        print("--- Training Complete ---")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()