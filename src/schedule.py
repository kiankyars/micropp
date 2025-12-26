import torch

def naive_pipeline_step(model, comms, batch, targets, hidden_dim, device):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.
    """
    
    # --- PHASE 1: FORWARD PASS ---
    
    # A. Get Input
    if comms.rank == 0:
        # Rank 0 gets the data directly from the dataloader
        input_data = batch.to(device)
    else:
        # Others wait to receive from the left
        batch_size = batch.shape[0] if batch is not None else 32 # Mock size
        shape = (batch_size, hidden_dim)
        input_data = comms.recv_forward(shape, device)
        # TEACHING MOMENT: In real PP, we need autograd to track this input tensor!
        input_data.requires_grad = True

    # B. Compute
    output = model(input_data, targets.to(device) if comms.rank == comms.world_size -1 else None)

    # C. Send Output
    if not model.is_last:
        comms.send_forward(output.detach()) # Send data to the right
        # We store 'output' and 'input_data' because we need them for backward pass
        
    # --- PHASE 2: BACKWARD PASS ---
    
    # A. Get Gradients
    if model.is_last:
        loss = output
        loss.backward() # This starts the chain reaction
        grad_to_send = input_data.grad 
    else:
        # Receive gradients coming from the right
        grad_from_next = comms.recv_backward(output.shape, device)
        
        # B. Compute Local Gradients
        # This is the "Backprop" step connecting the received grad to our weights
        output.backward(grad_from_next)
        grad_to_send = input_data.grad

    # C. Send Gradients
    if not model.is_first:
        comms.send_backward(grad_to_send)
        
    return loss.item() if model.is_last else None

def gpipe_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    GPipe Schedule: FWD all chunks -> BWD all chunks.
    """
    micro_batches = batch.chunk(chunks)
    batch_size = micro_batches[0].shape[0]
    shape = (batch_size, hidden_dim)
    
    # Storage for saved activations (needed for backward)
    input_buffers = [] 
    output_buffers = []
    
    # --- PHASE 1: ALL FORWARDS ---
    for i in range(chunks):
        # 1. Get Input
        if comms.rank == 0:
            mb = micro_batches[i].to(device)
        else:
            mb = comms.recv_forward(shape, device)
            mb.requires_grad = True # Critical for autograd!
            
        # 2. Forward
        # Use targets only if last chunk AND last GPU (simplification)
        current_target = targets if (model.is_last and i == chunks-1) else None
        
        # We assume the model returns 'output' (and loss is handled internally or separately)
        # For this edu-repo, let's say model always returns tensor, and we calc loss outside or inside.
        output = model(mb)
        
        # 3. Send
        if not model.is_last:
            comms.send_forward(output.detach())
        
        # 4. Cache for Backward
        input_buffers.append(mb)
        output_buffers.append(output)

    # --- PHASE 2: ALL BACKWARDS ---
    total_loss = 0
    
    for i in range(chunks):
        # We iterate normally, but logic applies to the specific buffered chunks
        # In real GPipe, we might iterate reverse, but here order matches buffers
        
        inp = input_buffers[i]
        out = output_buffers[i]
        
        if model.is_last:
            # Re-calculate loss for this micro-batch to get graph
            # (In production we'd cache the graph, but this is "Micro" style re-compute)
            # Simplification: Assume 'targets' is splittable or just use dummy for lab
            loss_val = out.mean() # Dummy loss for non-target micro-batches
            loss_val.backward()
            grad_to_send = inp.grad
            total_loss += loss_val.item()
        else:
            grad_from_next = comms.recv_backward(out.shape, device)
            out.backward(grad_from_next)
            grad_to_send = inp.grad
            
        if not model.is_first:
            comms.send_backward(grad_to_send)
            
    return total_loss / chunks