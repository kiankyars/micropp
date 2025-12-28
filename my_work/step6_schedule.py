from comms import PipelineComms
from model import ShardedMLP

def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms, batch, targets, hidden_dim, device):
    """
    A single training step using the Naive (Stop-and-Wait) schedule.

    TODOs:
    - Receive input from previous stage if not first stage
    - Forward microbatch through model
    - Send output to next stage if not last stage
    - Perform backward pass: 
        - If last stage, compute loss and call backward on it
        - Else, receive grad from next stage and call backward
    - Send grad to previous stage if not first stage
    - Return loss if last stage, else None
    """
    # TODO: If comms.rank == 0, use 'batch' directly; else, receive input
    # TODO: Forward pass through model
    # TODO: If not last stage, send output to next stage
    # TODO: Backward pass (different for last and non-last stage)
    # TODO: Send grad to previous stage if not first
    # TODO: Return loss if last stage, else None
    pass

def gpipe_pipeline_step(model, comms, batch, targets, hidden_dim, chunks, device):
    """
    GPipe Schedule: FWD all chunks -> BWD all chunks.
    """
    # TODO: Chunk the batch into microbatches
    # TODO: For i in [0..chunks):
    #     - If comms.rank == 0, use microbatch directly; else, receive input
    #     - Forward microbatch through model
    #     - If not last stage, send output to next stage
    #     - Append input/output to buffers
    # TODO: For i in [0..chunks):
    #     - Get inputs/outputs for this chunk from buffers
    #     - If last stage, compute loss and call backward
    #     - Else, receive grad from next stage and call backward
    #     - Send grad to previous stage if not first
    # TODO: Return loss if last stage, else None
    pass

# def 1f1b