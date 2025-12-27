from src.comms import PipelineComms
from src.model import ShardedMLP

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
    output = model(input_data, targets if comms.rank == comms.world_size -1 else None)

    # C. Send data to the right
    if not model.is_last:
        comms.send_forward(output.detach())
        
    # --- PHASE 2: BACKWARD PASS ---
    
    # A. Get Gradients
    if model.is_last:
        loss = output
        loss.backward() # This starts the chain reaction
        grad_to_send = input_data.grad 
    else:
        # Receive gradients coming from the right
        # whereas in the forward pass, we don't have the batch size unless we are the first device,
        # since gradients are propagated for every activation computed, we can just take
        # the activation dimensions for the shape of the gradient tensor we receive in backward()
        grad_from_next = comms.recv_backward(output.shape, device)
        # B. Compute Local Gradients
        # This is the "Backprop" step connecting the received grad to our weights
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

# def 1f1b