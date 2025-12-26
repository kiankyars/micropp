import torch

def naive_pipeline_step(model, comms, batch, targets, hidden_dim):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.
    """
    
    # --- PHASE 1: FORWARD PASS ---
    
    # A. Get Input
    if comms.rank == 0:
        # Rank 0 gets the data directly from the dataloader
        input_data = batch.cuda()
    else:
        # Others wait to receive from the left
        batch_size = batch.shape[0]
        shape = (batch_size, hidden_dim)
        input_data = comms.recv_forward(shape)
        # TEACHING MOMENT: In real PP, we need autograd to track this input tensor!
        input_data.requires_grad = True

    # B. Compute
    output = model(input_data, targets.cuda() if comms.rank == comms.world_size -1 else None)

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
        grad_from_next = comms.recv_backward(output.shape)
        
        # B. Compute Local Gradients
        # This is the "Backprop" step connecting the received grad to our weights
        output.backward(grad_from_next)
        grad_to_send = input_data.grad

    # C. Send Gradients
    if not model.is_first:
        comms.send_backward(grad_to_send)
        
    return loss.item() if model.is_last else None