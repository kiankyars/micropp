### Phase 1: The Learning & Teaching Syllabus

1. **Distributed Basics:**
* **Concept:** What is a Rank, World Size, and Process Group?
* **Tool:** `torch.distributed` (NCCL backend).
* **Lab:** Spawn 2 processes on 1 GPU (or CPU) and ping-pong a tensor.


2. **The Memory Wall:**
* **Concept:** Why models don't fit on one GPU.
* **Activity:** Calculate memory usage of a 10B param model vs. VRAM available.
* **Solution:** Model Partitioning (slicing `nn.Sequential`).


3. **The Naive Solution (and its failure):**
* **Concept:** Stop-and-wait execution.
* **Visual:** Draw a "Timeline Diagram" showing massive idle gaps (bubbles).
* **Lab:** Implement the Naive Schedule. Measure utilization (it will be low).


4. **The Pipelined Solution:**
* **Concept:** Micro-batches. Changing the loop from "Batch" to "Chunks."
* **Algorithms:** GPipe (Fill -> Drain) vs. 1F1B (Steady State).
* **Lab:** Implement Micro-batching. Watch utilization spike.



---

### Phase 2: Library Requirements

**1. `comms.py` (The Glue)**

* **Initialization:** A wrapper around `dist.init_process_group()`.
* **Topology:** `get_next_rank()` and `get_prev_rank()` based on current ID.
* **P2P:** `send_tensor(tensor, dest)` and `recv_tensor(shape, src)`.
* *Note:* Start with blocking communication (`dist.send`) for simplicity. Upgrade to async (`isend`) later if needed.



**2. `model.py` (The Subject)**

* **Deep MLP:** A class `DeepMLP(nn.Module)` with 16+ `Linear` layers.
* **Partitioner:** A function that takes a full model and returns only layers  to  based on the GPU rank.
* **Shape Consistency:** All layers must accept/return tensors of the exact same shape (simplifies the `recv` buffer allocation).

**3. `schedule.py` (The Engine)**

* **Micro-batcher:** A utility to `split` a batch tensor into  chunks.
* **The Orchestrator:** A loop that manages the "Clock Cycle":
* If Rank 0: Feed input chunk .
* If Rank : Calculate Loss.
* Else: Recv  Forward  Send.