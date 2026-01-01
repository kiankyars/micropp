import torch
from comms import PipelineComms
from model import ShardedMLP
from profiler import PipelineProfiler

def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, profiler: PipelineProfiler, batch, targets, hidden_dim, device):
    profiler.start_stage("step")
    
    with profiler.time_block("forward_get_input"):
        if comms.rank == 0:
            input_data = batch
        else:
            shape = (batch, hidden_dim)
            with profiler.time_block("forward_recv"):
                input_data = comms.recv_forward(shape, device)
            input_data.requires_grad = True
    
    with profiler.time_block("forward_compute"):
        output = model(input_data, targets if comms.rank == comms.world_size -1 else None)
    
    if not model.is_last:
        with profiler.time_block("forward_send"):
            comms.send_forward(output.detach())
    
    with profiler.time_block("backward_get_grad"):
        if model.is_last:
            loss = output
            with profiler.time_block("backward_compute"):
                loss.backward()
            grad_to_send = input_data.grad 
        else:
            with profiler.time_block("backward_recv"):
                grad_from_next = comms.recv_backward(output.shape, device)
            with profiler.time_block("backward_compute"):
                output.backward(grad_from_next)
            grad_to_send = input_data.grad
    
    if not model.is_first:
        with profiler.time_block("backward_send"):
            comms.send_backward(grad_to_send)
    
    profiler.end_stage("step")
    if model.is_last:
        return loss

def gpipe_pipeline_step(model: ShardedMLP, comms: PipelineComms, profiler: PipelineProfiler, batch, targets, hidden_dim, chunks, device):
    profiler.start_stage("step")
    
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)
    
    input_buffers = []
    output_buffers = []
    
    for i in range(chunks):
        with profiler.time_block("forward_get_input"):
            if comms.rank == 0:
                input_data = micro_batches[i]
            else:
                shape = (batch//chunks, hidden_dim)
                with profiler.time_block("forward_recv"):
                    input_data = comms.recv_forward(shape, device)
                input_data.requires_grad = True
        
        with profiler.time_block("forward_compute"):
            if comms.rank == comms.world_size - 1:
                output = model(input_data, micro_targets[i])
            else:
                output = model(input_data)
        
        if comms.rank != comms.world_size - 1:
            with profiler.time_block("forward_send"):
                comms.send_forward(output.detach())
        
        input_buffers.append(input_data)
        output_buffers.append(output)
    
    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(output.shape, device=device)
    
    for i in range(chunks):
        input_data = input_buffers[i]
        output = output_buffers[i]
        
        with profiler.time_block("backward_get_grad"):
            if comms.rank == comms.world_size - 1:
                loss = output / chunks
                with profiler.time_block("backward_compute"):
                    loss.backward()
                total_loss += loss
            else:
                with profiler.time_block("backward_recv"):
                    grad_from_next = comms.recv_backward(output.shape, device)
                with profiler.time_block("backward_compute"):
                    output.backward(grad_from_next)
        
        if comms.rank != 0:
            with profiler.time_block("backward_send"):
                comms.send_backward(input_data.grad)
    
    profiler.end_stage("step")
    if comms.rank == comms.world_size - 1:
        return total_loss

def onef_oneb_pipeline_step(model: ShardedMLP, comms: PipelineComms, profiler: PipelineProfiler, batch, targets, hidden_dim, chunks, device):
    profiler.start_stage("step")
    
    if comms.rank == 0:
        micro_batches = torch.chunk(batch, chunks)
    if comms.rank == comms.world_size - 1:
        micro_targets = targets.chunk(chunks)
    
    input_buffers = [None] * chunks
    output_buffers = [None] * chunks
    async_requests = []
    
    num_warmup = comms.world_size - comms.rank - 1
    num_1f1b = chunks - num_warmup
    
    def run_forward(micro_batch_idx):
        with profiler.time_block("forward_get_input"):
            if comms.rank == 0:
                input_data = micro_batches[micro_batch_idx]
            else:
                shape = (batch//chunks, hidden_dim)
                with profiler.time_block("forward_recv"):
                    input_data = comms.recv_forward(shape, device)
                input_data.requires_grad = True
        
        with profiler.time_block("forward_compute"):
            if comms.rank == comms.world_size - 1:
                output = model(input_data, micro_targets[micro_batch_idx])
            else:
                output = model(input_data)
        
        if comms.rank != comms.world_size - 1:
            with profiler.time_block("forward_send"):
                req = comms.isend_forward(output.detach())
                async_requests.append(req)
        
        input_buffers[micro_batch_idx] = input_data
        output_buffers[micro_batch_idx] = output
    
    def run_backward(micro_batch_idx):
        input_data = input_buffers[micro_batch_idx]
        output = output_buffers[micro_batch_idx]
        
        with profiler.time_block("backward_get_grad"):
            if comms.rank == comms.world_size - 1:
                loss = output / chunks
                with profiler.time_block("backward_compute"):
                    loss.backward()
            else:
                with profiler.time_block("backward_recv"):
                    grad_from_next = comms.recv_backward(output.shape, device)
                with profiler.time_block("backward_compute"):
                    output.backward(grad_from_next)
        
        if comms.rank != 0:
            with profiler.time_block("backward_send"):
                comms.send_backward(input_data.grad)
        
        if comms.rank == comms.world_size - 1:
            return loss
    
    if comms.rank == comms.world_size - 1:
        total_loss = torch.zeros(1, device=device)
    
    for i in range(num_warmup):
        run_forward(i)
    
    for i in range(num_1f1b):
        run_forward(i + num_warmup)
        res = run_backward(i)
        if comms.rank == comms.world_size - 1:
            total_loss += res
    
    for i in range(num_warmup):
        res = run_backward(i + num_1f1b)
        if comms.rank == comms.world_size - 1:
            total_loss += res
    
    profiler.end_stage("step")
    return total_loss if comms.rank == comms.world_size - 1 else None