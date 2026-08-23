# 04 — CSRNet: Dilated Convolution for Crowd Counting

**Paper:** *CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes* (Li et al., CVPR 2018)

**The problem CSRNet solves:** MCNN loses resolution through pooling and can't use pretrained features. CSRNet keeps full resolution via dilated convolutions and leverages a pretrained VGG16 backbone.

---

## 4.1 The Core Insight: Dilated (Atrous) Convolution

A normal 3×3 convolution looks at 9 adjacent pixels. A **dilated** 3×3 convolution has gaps between the pixels it samples — controlled by the **dilation rate**.

```
Normal 3×3 (rate=1):     Dilated 3×3 (rate=2):     Dilated 3×3 (rate=4):
X X X                     X . X . X                 X . . . X . . . X
X X X                     . . . . .                 . . . . . . . . .
X X X                     X . X . X                 . . . . . . . . .
                          . . . . .                 X . . . X . . . X
                          X . X . X                 . . . . . . . . .
                                                   . . . . . . . . .
                                                   X . . . X . . . X

Receptive field: 3×3      Receptive field: 7×7      Receptive field: 15×15
Parameters: 9             Parameters: 9              Parameters: 9
```

**Key property:** the receptive field grows quadratically with dilation rate, but the number of parameters stays the same (9 weights for a 3×3 kernel).

### Why this matters for crowd counting

| Normal conv | Dilated conv |
|---|---|
| To cover a 15×15 area: use a 15×15 kernel (225 params) or stack 7× 3×3 convs | Use one 3×3 conv with rate=4 (9 params) |
| Each pooling layer halves resolution | No pooling needed — resolution stays the same |
| Deep networks need many layers to see the whole image | A few dilated layers can see the whole image |

---

## 4.2 CSRNet Architecture

```
Input (H×W×3)
│
├── VGG16 Frontend (conv layers only, no dense)
│   ├── Block 1: Conv(3→64) ×2 + ReLU + MaxPool(2)    → H/2 × W/2 × 64
│   ├── Block 2: Conv(64→128) ×2 + ReLU + MaxPool(2)   → H/4 × W/4 × 128
│   ├── Block 3: Conv(128→256) ×3 + ReLU + MaxPool(2)  → H/8 × W/8 × 256
│   └── Block 4: Conv(256→512) ×3 + ReLU               → H/8 × W/8 × 512
│       ↑ NO POOLING after block 4 — resolution stays at H/8
│
├── Dilated Backend (trained from scratch)
│   ├── Conv(512→512, 3×3, dilation=2, padding=2) + ReLU
│   ├── Conv(512→512, 3×3, dilation=2, padding=2) + ReLU
│   ├── Conv(512→512, 3×3, dilation=2, padding=2) + ReLU
│   ├── Conv(512→256, 3×3, dilation=2, padding=2) + ReLU
│   ├── Conv(256→128, 3×3, dilation=2, padding=2) + ReLU
│   └── Conv(128→64, 3×3, dilation=2, padding=2) + ReLU
│
└── Output
    └── Conv(64→1, 1×1) → Density map (H/8 × W/8 × 1)
```

### Why stop at conv4 and remove pooling?

VGG16 normally has 5 blocks with pooling after each. CSRNet:
- Keeps blocks 1-3 with pooling (reduces 512×512 → 64×64)
- **Removes pooling after block 4** — resolution stays at 64×64 instead of dropping to 32×32
- Replaces block 5 (which would have more pooling) with dilated convs

**Result:** output is 1/8 of input size, not 1/32. Much sharper density maps.

---

## 4.3 Receptive Field Comparison

Let's trace the receptive field at each stage for a 512×512 input:

| Layer | Cumulative RF (CSRNet) | Cumulative RF (standard VGG) |
|---|---|---|
| After block 1 | 10×10 | 10×10 |
| After block 2 | 30×30 | 30×30 |
| After block 3 | 70×70 | 70×70 |
| After block 4 | 150×150 | 150×150 |
| After block 5 (pool) | — | 366×366 (but 16×16 map) |
| After dilated backend | **510×510** (covers almost entire 512×512 image) | — |

CSRNet's dilated backend covers nearly the whole image while keeping a **64×64 feature map**. Standard VGG would have a 16×16 map — 16× less spatial information.

---

## 4.4 Why Pretrained VGG16 Matters

The VGG16 frontend is **pretrained on ImageNet** (1.2 million images, 1000 classes). This gives CSRNet:

| Benefit | Why it helps |
|---|---|
| Rich low-level features | Edges, textures, colors — already learned, no need to rediscover |
| Faster convergence | Start from good weights, not random |
| Less data needed | Fine-tune with crowd data instead of training from scratch |
| Better generalization | ImageNet features transfer well to crowd scenes |

**What gets fine-tuned:** the VGG frontend weights are updated during training (fine-tuned), but they start from a much better place than random initialization.

**What's trained from scratch:** the dilated backend and the 1×1 output conv.

---

## 4.5 CSRNet in Code (PyTorch Sketch)

```python
import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained VGG16, take only the feature extractor
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.frontend = vgg.features[:23]  # Up to conv4 (block 4, layer 3)

        # Dilated backend
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
        )

        # Output layer
        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)   # (B, 512, H/8, W/8)
        x = self.backend(x)    # (B, 64, H/8, W/8)
        x = self.output(x)     # (B, 1, H/8, W/8)
        return x
```

**Note:** `vgg.features[:23]` gives the first 23 layers of VGG16's feature extractor, which corresponds to everything up to and including conv4_3 (the third conv in block 4), but **without** the maxpool after it. In practice you may need to slice carefully to exclude that pool.

---

## 4.6 Reference Training Details

The following table records the paper-aligned target configuration, not the
complete behavior of the repository's current generic training entrypoint.

| Hyperparameter | Value |
|---|---|
| Loss | Pixel-wise MSE on density map |
| Optimizer | SGD with momentum (0.9) |
| Learning rate | 1e-6 (frontend), 1e-5 (backend) — lower LR for pretrained layers |
| Weight decay | 5e-4 |
| Batch size | 1 (due to memory — large images + many channels) |
| Data aug | Random crops, horizontal flips, color jitter |
| Ground truth | Fixed Gaussian kernel (sigma=4 for ShanghaiTech Part A, sigma=15 for Part B) |

**End-to-end fine-tuning:** VGG16 frontend starts from ImageNet weights, while
the dilated backend and output layer start from newly initialized weights. All
layers remain trainable.

The current `src/train.py` uses one SGD parameter group and the single CSRNet
learning rate from `MODEL_CONFIGS`; it does not yet apply separate frontend and
backend learning rates or weight decay. Those paper-aligned optimizer details
are intentionally deferred to the CSRNet training-integration task.

---

## 4.7 Strengths and Weaknesses

### Strengths

| Strength | Why |
|---|---|
| High-resolution output | No pooling after conv4 → 1/8 resolution instead of 1/32 |
| Pretrained features | VGG16 backbone gives rich features from the start |
| Large receptive field | Dilated convs cover the whole image |
| Single column | Simpler and faster than MCNN's three columns |
| State-of-the-art (2018) | Significantly outperformed MCNN on ShanghaiTech, UCF, etc. |

### Weaknesses

| Weakness | Why |
|---|---|
| Gridding artifacts | Dilated convs with the same rate stacked → some pixels never sampled |
| Fixed dilation rates | Rate=2 for all backend layers — not adaptive to different scales |
| Memory hungry | High-resolution feature maps (64×64×512) need lots of GPU memory |
| VGG is old | VGG16 is from 2014 — newer backbones (ResNet, HRNet) would be better |

---

## 4.8 The Gridding Artifact Problem

Stacking dilated convs with the same dilation rate creates a problem: some pixels are never sampled.

```
Dilated 3×3, rate=2, stacked 3 times:
Layer 1:  X . X . X
          . . . . .
          X . X . X
          . . . . .
          X . X . X

Layer 2:  Same pattern on the output of layer 1
          → information from the "gaps" never propagates

Layer 3:  Even worse — large regions are blind
```

**Solutions (from later papers):**
- Use **increasing dilation rates** (1, 2, 3, 4 instead of 2, 2, 2, 2)
- Use **hybrid dilated convolution (HDC)** — mix rates to cover all pixels
- Use **deformable convolution** — learn the offsets instead of fixed dilation

CSRNet uses rate=2 for all backend layers, which is suboptimal but still works well in practice.
