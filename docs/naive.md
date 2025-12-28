### Naive Model Parallelism

- Naive model parallelism is the most straightforward way of implementing pipeline-parallel training.
- We split our model into multiple parts, and assign each one to a GPU.
- Then we train, inserting communication steps at the boundaries where we’ve split the model.
- We only use node-to-node communication (MPI.Send and MPI.Recv) and don’t need any collective communication primitives.
- ![](https://lilianweng.github.io/posts/2021-09-25-train-large/naive-data-parallelism.png)

#### Forward Pass

1. Compute intermediate on GPU1.
2. Transfer the resulting tensor to GPU2.
3. GPU2 computes the loss of the model.

#### Backward Pass

1. GPU2 calculates derivative of loss w.r.t its weights and input.
2. Send the gradients w.r.t. intermediate from GPU2 to GPU1.
3. GPU1 then completes the backward pass based on the gradients it was sent.

![](./PP_pebble_graph.gif)

By looking at the pebble graph, we can observe some inefficiencies of naive model parallelism:

1. Low GPU utilization.
2. No interleaving of communication and computation.
3. High memory demand.
