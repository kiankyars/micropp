from comms import PipelineComms
from model import ShardedMLP

def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, batch, targets, hidden_dim, device):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.
    """
    if comms.rank == 0:
        input_data = batch
    else:
        shape = (batch, hidden_dim)
        input_data = comms.recv_forward(shape, device)
        input_data.requires_grad = True

    output = model(input_data, targets)

    if not model.is_last:
        comms.send_forward(output.detach())
        
    if model.is_last:
        loss = output
        loss.backward()
    else:
        grad_from_next = comms.recv_backward(output.shape, device)
        output.backward(grad_from_next)
    grad_to_send = input_data.grad
        
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
    
    input_buffers = [] 
    output_buffers = []
    
    for i in range(chunks):
        if comms.rank == 0:
            mb = micro_batches[i].to(device)
        else:
            mb = comms.recv_forward(shape, device)
            mb.requires_grad = True
            
        current_target = targets if (model.is_last and i == chunks-1) else None
        output = model(mb)
        
        if not model.is_last:
            comms.send_forward(output.detach())
        
        input_buffers.append(mb)
        output_buffers.append(output)

    total_loss = 0
    
    for i in range(chunks):
        inp = input_buffers[i]
        out = output_buffers[i]
        
        if model.is_last:
            loss_val = out.mean()
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