### Syllabus

#### [Monolith](https://github.com/kiankyars/micropp/blob/main/src/monolith.py):

Start with a MLP and train it on a single CPU.

- **The Goal:** Establish a baseline.
- **The Code:** Just `nn.Sequential` with 16 layers and a simple training loop.

#### Motivation for PP:

- **The Memory Wall:** Why models don't fit on one GPU.
  - **Model Size:** 10 Billion Parameters (10B).
  - **Precision:** FP32 (4 bytes per parameter).
  - **VRAM:** 40 GB.
  - **Hardware:** NVIDIA RTX 4090 (24 GB VRAM).
- **Solution:** Model Partitioning (slicing `nn.Sequential`).

#### [Manual](https://github.com/kiankyars/micropp/blob/main/src/manual.py):

Cut the `nn.Sequential` into two pieces: `part1` and `part2`.

- **The Exercise:** Try to train it by manually passing the output of `part1` into `part2`.
- **The Lesson:** Even on one machine, you have to manage the "hand-off" of the activation and the gradient.

#### Distributed Basics:

- **Concept:** What is a Rank, World Size, and Process Group?
  - **The Process Group:** Imagine a conference call. Before anyone can talk, they must dial in. `init_process_group` is dialing in.
  - **World Size:** The total number of people on the call (e.g., 4 GPUs).
  - **Rank:** Your unique ID badge (0, 1, 2, 3).
  - **Rank 0** is the "Boss" (usually handles logging, saving checkpoints, and data loading).
- **Lab:** Spawn 2 processes on GPU (or CPU) and ping-pong a tensor.
  - _Run command:_ `torchrun --nproc_per_node=2 lab_pingpong.py`


#### [The Naive Solution](./naive.md):

- **Concept:** Stop-and-wait execution.
- **Lab:** Implement the Naive Schedule. Measure utilization using nvidia-smi if cuda is available.

#### [GPipe](./gpipe.md):

- **Concept:** Changing the loop from "Batch" to "Chunks."
- **Lab:** GPipe (Fill -> Drain).

#### [1F1B](./1f1b.md):

- **Concept:** Interleaving Chunks.
- **Lab:** 1F1B (Steady State).

---

### Library

**1. `comms.py` (The Glue)**

- **Initialization:** A wrapper around `dist.init_process_group()`.
- **Topology:** `get_next_rank()` and `get_prev_rank()` based on rank.
- **P2P:** `send_tensor(tensor, dest)` and `recv_tensor(shape, src)`.

**2. `model.py` (The Subject)**

- **ShardedMLP:** A class `ShardedMLP(nn.Module)` with 16 `Linear` layers.

**3. `schedule.py` (The Engine)**

- **Micro-batcher:** A utility to `split` a batch tensor into chunks.
- **The Orchestrator:** A loop that manages the "Clock Cycle".
