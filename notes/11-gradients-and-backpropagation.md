# 11 — Gradients and Backpropagation

A **gradient** is the multi-dimensional generalization of a derivative. It tells you how to nudge each parameter to reduce the loss. Backpropagation is the algorithm that computes all gradients automatically.

---

## 11.1 Derivatives — The 1D Case

For a function of one variable, the derivative tells you "how does the output change when I nudge the input?"

```
f(x) = 3x²

f'(x) = 6x     ← derivative: one number per input
```

At `x=2`: `f'(2) = 12`. If you nudge x by +0.01, output goes up by ~0.12.

**Key property:** the derivative points in the direction of steepest **increase**. To **minimize** the function, move in the **opposite** direction:

```
x_new = x_old - learning_rate × f'(x_old)
```

---

## 11.2 Gradients — The Multi-D Case

For a function of multiple variables, the gradient is a **vector of partial derivatives** — one derivative per input variable:

```
f(x, y) = 3x² + 2y³

∂f/∂x = 6x     ← how f changes when x moves (holding y fixed)
∂f/∂y = 6y²    ← how f changes when y moves (holding x fixed)

gradient = [6x, 6y²]    ← a vector, not a single number
```

At `(x=2, y=3)`:

```
gradient = [12, 54]
```

The gradient points in the **direction of steepest increase** in the multi-dimensional space. To decrease the function, move in the opposite direction — this is **gradient descent**.

---

## 11.3 Gradients in Neural Networks

The loss function depends on **every weight** in the network — potentially millions:

```
loss = f(w₁, w₂, w₃, ..., w₅₀₀₀₀₀₀)
```

The gradient is a vector with one entry per weight:

```
gradient = [∂loss/∂w₁, ∂loss/∂w₂, ..., ∂loss/∂w₅₀₀₀₀₀]

shape: (500000,)  ← same shape as the weights
```

Each entry tells you: "if I nudge this weight slightly, how much does the loss change?"

### The training loop in one equation

```
new_weight = old_weight - learning_rate × gradient
```

Move each weight slightly in the direction that **decreases** the loss. That's it. All of deep learning training is this one equation, repeated millions of times.

```
gradient = [∂loss/∂w₁, ∂loss/∂w₂, ...]
weights -= learning_rate × gradient
```

---

## 11.4 The Chain Rule

Neural networks are **composed functions** — the output of one layer is the input to the next:

```
image → conv1 → conv2 → conv3 → loss
```

To get `∂loss/∂w₁` (how loss changes w.r.t. the first layer's weights), you need the **chain rule**:

```
∂loss/∂w₁ = ∂loss/∂conv3 × ∂conv3/∂conv2 × ∂conv2/∂conv1 × ∂conv1/∂w₁
```

Each layer contributes one link in the chain. Backpropagation computes this by walking **backward** from the loss to the input, multiplying local derivatives at each step.

This is why it's called **back**propagation — the gradient flows backward through the network:

```
Forward:  image → layer1 → layer2 → layer3 → loss
Backward: loss → ∂loss → ∂layer3 → ∂layer2 → ∂layer1 → ∂weights
```

---

## 11.5 Automatic Differentiation in PyTorch

PyTorch computes gradients automatically. You don't derive them by hand.

### The three-step recipe

```python
# 1. Tell PyTorch to track gradients on the weights
w = torch.tensor([2.0], requires_grad=True)

# 2. Forward pass: compute the output
y = 3 * w ** 2          # y = 3w²

# 3. Backward pass: compute gradients
y.backward()            # PyTorch applies the chain rule automatically

# Result
w.grad                  # tensor([12.])  ← ∂y/∂w = 6w = 6×2 = 12
```

### What each line does

**`requires_grad=True`** — "please remember all operations performed on this tensor, so I can compute gradients later." PyTorch builds a **computation graph** behind the scenes:

```
w (leaf, requires_grad=True)
  → mul(3) → temp1
  → pow(2) → y (output)
```

**`y.backward()`** — "walk backward through the computation graph and compute the gradient of y with respect to every tensor that has `requires_grad=True`." For our example:

```
y = 3w²
dy/dw = 6w = 6 × 2 = 12  → stored in w.grad
```

**`w.grad`** — stores the gradient. For a weight tensor of shape `(500000,)`, `w.grad` is also shape `(500000,)` — the gradient for every weight.

---

## 11.6 The Full Training Loop

```python
# Setup
model = CSRNet()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# Training loop
for images, density_maps in dataloader:
    # 1. Forward pass
    predictions = model(images)              # image → prediction
    loss = loss_fn(predictions, density_maps)  # prediction, target → loss

    # 2. Backward pass
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute new gradients (chain rule, automatically)

    # 3. Update weights
    optimizer.step()        # w -= lr × gradient, for every weight
```

### Why `optimizer.zero_grad()`?

PyTorch **accumulates** gradients by default. If you don't clear them, each batch's gradients add to the previous batch's. This is rarely what you want — you want fresh gradients for each batch.

```python
# Without zero_grad:
# Batch 1: w.grad = 12
# Batch 2: w.grad = 12 + 18 = 30  ← wrong! mixed gradients
# Batch 3: w.grad = 30 + 15 = 45  ← even worse

# With zero_grad:
# Batch 1: w.grad = 12
# Batch 2: w.grad = 18  ← correct
# Batch 3: w.grad = 15  ← correct
```

---

## 11.7 What `backward()` Actually Does

When you call `loss.backward()`, PyTorch walks backward through the computation graph:

```
loss = MSE(prediction, target)
  prediction = conv_backend(conv_frontend(image))
    conv_backend = dilated_conv(conv_frontend_output)
    conv_frontend = vgg16_layers(image)

Backward:
  ∂loss/∂prediction          ← from MSE
  × ∂prediction/∂conv_backend ← from dilated_conv
  × ∂conv_backend/∂conv_frontend  ← from dilated_conv
  × ∂conv_frontend/∂weights       ← from VGG layers
  = ∂loss/∂weights                ← stored in weight.grad
```

Each operation in the forward pass registers a **backward function**. `backward()` calls them in reverse order, passing gradients through the chain rule.

This is **automatic differentiation** — you write the forward pass (just the model and loss), and PyTorch derives and runs the backward pass for you.

---

## 11.8 Why Tensors Need Gradients

This is the entire reason `torch.Tensor` exists separately from `numpy.array`:

| | numpy array | torch tensor |
|---|---|---|
| Multi-dimensional array | ✅ | ✅ |
| GPU support | ❌ | ✅ |
| Gradient tracking | ❌ | ✅ (`requires_grad`) |
| Automatic differentiation | ❌ | ✅ (`backward()`) |
| Optimizer integration | ❌ | ✅ (`optimizer.step()`) |

Without gradient tracking, you'd have to manually derive and implement the backward pass for every layer — which is what researchers did before frameworks like PyTorch/TF existed. Autograd made deep learning practical at scale.

---

## 11.9 Summary

| Concept | Key point |
|---|---|
| **Derivative** | How a function changes w.r.t. one variable |
| **Gradient** | Vector of partial derivatives — one per parameter |
| **Chain rule** | How to differentiate composed functions (layer stacks) |
| **Backpropagation** | Walk backward through the network, applying chain rule |
| **`requires_grad=True`** | Tell PyTorch to track operations on this tensor |
| **`backward()`** | Compute all gradients automatically |
| **`w.grad`** | Where the gradient is stored |
| **`optimizer.step()`** | Apply `w -= lr × gradient` to all weights |
| **`zero_grad()`** | Clear old gradients before each batch |