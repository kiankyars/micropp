### Scenario 1: With `requires_grad = True` (Working Pipeline)

In this case, we allow the "chain" of math to stay connected across the two devices.

1. **Rank 0** performs a calculation (A×B=C) and sends the result to **Rank 1**.
2. **Rank 1** receives C. We manually set `C.requires_grad = True`.
3. **Rank 1** performs its own calculation (C×D=Output).
4. When we call `.backward()` on Rank 1, PyTorch calculates the gradient for its weights (D) **and** for the input C.
5. **The Result:** Rank 1 now has a value for `C.grad`. It calls `send_backward(C.grad)` to Rank 0. Rank 0 receives this and can now calculate the gradients for its own weights (B).

### Scenario 2: Without `requires_grad = True` (Broken Pipeline)

In this case, Rank 1 treats the incoming data as a "constant" rather than a variable.

1. **Rank 0** sends C to **Rank 1**.
2. **Rank 1** receives C. By default, `requires_grad` is **False**.
3. **Rank 1** calculates (Output=C×D).
4. When we call `.backward()` on Rank 1, PyTorch calculates the gradient for the weights (D). However, because C was marked as a constant (no grad required), the engine **stops** there. 🛑
5. **The Result:** `C.grad` is `None`. Rank 1 has nothing to send back. Rank 0 never receives a gradient, so its weights (B) never move. **Only half the model learns.** ❌

Essentially, `requires_grad = True` creates a "hook" at the very edge of the device's memory. Without that hook, the backward pass has nothing to grab onto to pull the information back across the network to the other device.
