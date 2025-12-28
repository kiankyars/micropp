### GPipe

![](https://1.bp.blogspot.com/-fXZxDPKaEaw/XHlt7OEoMtI/AAAAAAAAD0I/hYM_6uq2BTwaunHZRxWd7JUJV43fEysvACLcBGAs/s640/image2.png)

#### Interleaved GPipe

![](./interleaved-GPipe.png)

#### Bubbles

- ![](./bubble.png)

- Increasing the number of microbatches m, is necessary for making the bubble fraction small; increasing the memory demand.
  As the number of micro-batches (m) increases relative to the number of GPUs (n), the bubble fraction (
  m
  n−1
  ​
  ) shrinks. This is why "flooding the pipe" with many small micro-batches is the key to high hardware efficiency in systems like GPipe.
  ![](https://siboehm.com/assets/img/distributed-DNNs/Gpipe_bubble_fractions.png)

#### Memory Demand

![](https://siboehm.com/assets/img/distributed-DNNs/GPipe-gradient-checkpointing.png)
