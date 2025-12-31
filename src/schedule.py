import torch
from comms import PipelineComms
from model import ShardedMLP

def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, batch, targets, hidden_dim, device):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.
    """
    
    # --- PHASE 1: FORWARD PASS ---
    
    # A. Get Input
    if comms.rank == 0:
        # Rank 0 gets the data directly from the dataloader
        input_data = batch
    else:
        # we need the shape here, but not for above because 
        # the first device just gets the data straight from 
        # the data loader and doesn't need to make a buffer
        # tensor to receive the activations
        shape = (batch, hidden_dim)
        # Others wait to receive from the left
        input_data = comms.recv_forward(shape, device)
        # When we receive a tensor via dist.recv, it is a leaf node
        # By default, leaf nodes have requires_grad = False.
        # Because we set requires_grad = True, the engine
        # continues the chain rule all the way back to that input tensor
        # Without this, the tensor would be treated as a constant by autograd and
        # input_data.grad would be set to None in backward(),
        # so earlier (upstream) model parameters wouldn’t update
        input_data.requires_grad = True

    # B. Compute
    # if you are not last, you just calculate activations
    # if you are last, you also calculate the loss with targets, which is output
    output = model(input_data, targets)

    # C. Send data to the right
    if not model.is_last:
        comms.send_forward(output.detach())
        
    # --- PHASE 2: BACKWARD PASS ---
    
    # A. Get Gradients
    if model.is_last:
        loss = output
        # Scalar Backward: loss.backward() (Used only on the last GPU).
        # Non-Scalar Backward: output.backward(gradient) (Used on all previous GPUs).
        loss.backward() # This starts the chain reaction
    else:
        # Receive gradients coming from the right
        # whereas in the forward pass, we don't have the batch size unless we are the first device,
        # since gradients are propagated for every activation computed, we can just take
        # the activation dimensions for the shape of the gradient tensor we receive in backward()
        grad_from_next = comms.recv_backward(output.shape, device)
        # B. Compute Local Gradients
        # This is the "Backprop" step connecting the received grad to our weights
        # When you call .backward() on a non-scalar tensor (like a hidden activation with
        # shape [32, 128]), PyTorch requires a "matching" gradient tensor of the same shape.
        # This provided gradient acts as the starting point for the Vector-Jacobian Product,
        # allowing the chain rule to flow backward to the weights and the input.
        output.backward(grad_from_next)
    grad_to_send = input_data.grad
    '''
    ∂Weights/∂Loss are the gradients which tell the model how to change its own internal layers.
    ∂Input/∂Loss are the gradients which we back-propagate; if Rank 0 is the very first layer
    (taking in the raw data/images), it technically calculates the gradient with respect to
    the raw input, but we discard this because we can't "update" the training data!
    '''
    # C. Send Gradients
    if not model.is_first:
        comms.send_backward(grad_to_send)
    if model.is_last:
        return loss

def gpipe_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    GPipe Schedule: FWD all chunks -> BWD all chunks.
    """
    # 1. Prepare Data Slices
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)
    
    # Storage for "Phase 2"
    input_buffers = [] 
    output_buffers = []
    
    # --- PHASE 1: ALL FORWARDS (Fill the Pipe) ---
    for i in range(chunks):
        # A. Setup Input
        if comms.rank == 0:
            input_data = micro_batches[i]
        else:
            shape = (batch//chunks, hidden_dim)
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True

        # B. Forward Pass
        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[i])
        else:
            output = model(input_data)
            comms.send_forward(output.detach())

        # D. Buffer for Backward
        input_buffers.append(input_data)
        output_buffers.append(output) # On last rank, this is the Loss

    # --- PHASE 2: ALL BACKWARDS (Drain the Pipe) ---
    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(output.shape)
    # Layers: Reverse Order (handled by Autograd).
    # Micro-batches: Forward Order (handled by loop to match the send/recv order).
    # Think of a conveyor belt
    # Both loop orders give the same result here because each micro-batch's 
    # forward and backward passes are fully independent of the others in this 
    # GPipe schedule: all forwards are completed and stored before any backward 
    # begins, so the order of backward iteration (reversed or not) does not 
    # change gradients or loss accumulation across chunks.
    for i in range(chunks):
        # Retrieve state from Phase 1
        input_data = input_buffers[i]
        output = output_buffers[i]
        
        if comms.rank == comms.world_size - 1:
            # On Last Rank, 'output' IS the loss
            loss = output / chunks
            loss.backward()
            total_loss += loss
        else:
            # On other ranks, we need gradients from downstream
            grad_from_next = comms.recv_backward(output.shape, device)
            output.backward(grad_from_next)
            
        # Send gradients backward (if not first)
        if comms.rank != 0:
            comms.send_backward(input_data.grad)
            
    # Return loss across chunks (for logging) if last rank
    if comms.rank == comms.world_size - 1:
        return total_loss

def onef_oneb_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    1F1B Schedule: Interleaves Forward and Backward passes.
    """
        # 1. Prepare Data Slices
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)
    
    # Storage for "Phase 2"
    input_buffers = [None] * chunks 
    output_buffers = [None] * chunks
    async_requests = [] # Keep request objects alive to prevent buffer deallocation
    
    # Schedule Logic
    # Rank 0 (First) has max warmup (needs to fill the whole pipe)
    # Rank N (Last) has 0 warmup (can backward immediately)
    num_warmup = comms.world_size - comms.rank - 1
    num_1f1b = chunks - num_warmup
    
    def run_forward(micro_batch_idx):
        # ... Setup Input ...
        if comms.rank == 0:
            input_data = micro_batches[micro_batch_idx]
        else:
            shape = (batch//chunks, hidden_dim)
            input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True

        # B. Forward Pass
        if comms.rank == comms.world_size - 1:
            output = model(input_data, micro_targets[micro_batch_idx])
        else:
            output = model(input_data)
            # Store request to keep tensor buffer alive (prevents GC before Gloo finishes)
            req = comms.isend_forward(output.detach())
            async_requests.append(req)

        input_buffers[micro_batch_idx] = input_data
        output_buffers[micro_batch_idx] = output

    def run_backward(micro_batch_idx):
        input_data = input_buffers[micro_batch_idx]
        output = output_buffers[micro_batch_idx]
        
        if comms.rank == comms.world_size - 1:
            loss = output / chunks
            loss.backward()
        else:
            grad_from_next = comms.recv_backward(output.shape, device)
            output.backward(grad_from_next)
            
        if comms.rank != 0:
            # Store the async request handle
            comms.send_backward(input_data.grad)
        
        if comms.rank == comms.world_size - 1:
            return loss

    # --- EXECUTION PHASES ---
    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(1, device=device)

    # Phase 1: Warmup (Forward Only)
    for i in range(num_warmup):
        run_forward(i)

    # Phase 2: Steady State (1F1B)
    for i in range(num_1f1b):
        run_forward(i + num_warmup)
        # run_backward returns the loss (on last rank) or None (others)
        res = run_backward(i)
        if comms.rank == comms.world_size - 1:
            total_loss += res

    # Phase 3: Cooldown (Backward Only)
    for i in range(num_warmup):
        res = run_backward(i + num_1f1b)
        if comms.rank == comms.world_size - 1:
            total_loss += res
    
    # Return Loss
    return total_loss if comms.rank == comms.world_size - 1 else None