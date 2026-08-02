# 03 — MCNN: Multi-Column CNN

**Paper:** *Single Image Crowd Counting via Multi-Column Convolutional Neural Network* (Zhang et al., CVPR 2016)

**The problem MCNN solves:** heads in a crowd appear at vastly different scales depending on their distance from the camera. A single CNN with one filter size can't capture all scales well.

---

## 3.1 The Scale Problem

In a single crowd image:

```
Nearby head:  ~30×30 pixels     Far head:  ~5×5 pixels
┌──────────────┐                 ┌─────┐
│  ●  ●  ●     │                 │ ●  │
│  ●  ●  ●     │                 │    │
│  ●  ●  ●     │                 └─────┘
└──────────────┘
```

A 3×3 filter sees 9 pixels. On a 30×30 head, that's a tiny patch of hair — useless. On a 5×5 head, that's the whole head — perfect.

A 9×9 filter sees 81 pixels. On a 30×30 head, that's a meaningful region. On a 5×5 head, that's way too much context.

**No single filter size works for both.**

---

## 3.2 MCNN's Solution: Three Parallel Columns

Run three independent conv columns in parallel, each with different filter sizes, then merge.

```
                    ┌─────────────────────────────────────┐
                    │  Column 1 (large filters)            │
Input (H×W×3) ────→│  Conv(9×9, 3→16) → Pool(2×2)        │
                    │  Conv(7×7, 16→32) → Pool(2×2)       │
                    └─────────────────────────────────────┘
                    │
                    ├──→ Column 2 (medium filters)         │
                    │  Conv(7×7, 3→20) → Pool(2×2)        │
                    │  Conv(5×5, 20→40) → Pool(2×2)       │
                    └─────────────────────────────────────┘
                    │
                    └──→ Column 3 (small filters)          │
                       Conv(5×5, 3→24) → Pool(2×2)        │
                       Conv(3×3, 24→48) → Pool(2×2)       │
                    └─────────────────────────────────────┘
                                        │
                                        ▼
                              Concat (channel-wise)
                              H/4 × W/4 × (32+40+48=120)
                                        │
                                        ▼
                              Conv(1×1, 120→1) → Density map
```

### Column details

| Column | Filter sizes | Channels | Receptive field | Best for |
|---|---|---|---|---|
| 1 (large) | 9×9 → 7×7 | 16 → 32 | Large | Nearby heads |
| 2 (medium) | 7×7 → 5×5 | 20 → 40 | Medium | Mid-distance heads |
| 3 (small) | 5×5 → 3×3 | 24 → 48 | Small | Far heads |

**Why different channel counts?** The small-filter column has more channels (24→48) because small filters have fewer parameters per filter — so it can afford more of them. The large-filter column has fewer channels (16→32) because each filter is expensive.

---

## 3.3 Channel-wise Concatenation

After the two conv+pool blocks in each column, all three outputs are concatenated along the channel dimension:

```
Column 1 output:  H/4 × W/4 × 32
Column 2 output:  H/4 × W/4 × 40
Column 3 output:  H/4 × W/4 × 48
                              ↓
Concatenated:     H/4 × W/4 × 120
```

Then a 1×1 conv fuses the 120 channels into 1 — the density map.

**Why concat and not average/sum?** Concatenation preserves all information from all columns. The 1×1 conv learns *how* to weight each column's output at each spatial position. In dense crowds, it might favor the small-filter column. In sparse crowds, the large-filter column.

---

## 3.4 Training Details

- **Loss:** Pixel-wise MSE between predicted and ground truth density map
- **Optimizer:** SGD with momentum
- **Learning rate:** 1e-6 (very small — MCNN is trained from scratch, no pretraining)
- **Data augmentation:** Random crops, horizontal flips
- **Ground truth:** Geometry-adaptive Gaussian kernels (sigma proportional to nearest neighbor distance)

### Why no pretraining?

MCNN's columns use non-standard filter sizes (9×9, 7×7, 5×5). There's no pretrained model with these architectures. Everything must be learned from scratch, which means:
- Needs more data
- Takes longer to converge
- Lower quality features than pretrained alternatives

This is MCNN's biggest weakness.

---

## 3.5 Strengths and Weaknesses

### Strengths

| Strength | Why |
|---|---|
| Handles scale variation | Three columns explicitly designed for different scales |
| Simple idea | Easy to understand and implement |
| Pioneering | First deep learning approach to crowd counting |

### Weaknesses

| Weakness | Why |
|---|---|
| No pretraining | All weights learned from scratch → needs lots of data |
| Expensive | Three columns = ~3× computation of a single-column network |
| Pooling loses resolution | Two pooling layers → output is 1/4 of input size |
| Fixed scales | Three fixed filter sizes — what if the scale distribution doesn't match? |
| No shared features | Each column learns edges/textures independently — wasteful |

---

## 3.6 MCNN in Code (PyTorch Sketch)

```python
import torch
import torch.nn as nn

class MCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Column 1: large filters
        self.col1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Column 2: medium filters
        self.col2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Column 3: small filters
        self.col3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fusion: 1×1 conv to produce density map
        self.fusion = nn.Sequential(
            nn.Conv2d(32 + 40 + 48, 1, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.col1(x)   # (B, 32, H/4, W/4)
        out2 = self.col2(x)   # (B, 40, H/4, W/4)
        out3 = self.col3(x)   # (B, 48, H/4, W/4)
        out = torch.cat([out1, out2, out3], dim=1)  # (B, 120, H/4, W/4)
        out = self.fusion(out)  # (B, 1, H/4, W/4)
        return out
```

---

## 3.7 When to Use MCNN

- You have a **lot of training data** (no pretraining available)
- The crowd has **extreme scale variation** (both very near and very far heads)
- You want a **simple, interpretable** architecture
- You don't need **real-time** inference (three columns are slow)

MCNN was state-of-the-art in 2016. CSRNet (2018) improved on it significantly — see the next file.
