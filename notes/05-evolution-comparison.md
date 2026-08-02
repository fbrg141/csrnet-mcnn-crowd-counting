# 05 — Evolution & Comparison: CNN → MCNN → CSRNet

A side-by-side comparison of the three architectures, the evolution of ideas, and guidance on when to use each.

---

## 5.1 Architecture Comparison

| Aspect | Standard CNN | MCNN | CSRNet |
|---|---|---|---|
| **Year** | — | 2016 | 2018 |
| **Task** | Classification | Crowd counting | Crowd counting |
| **Output** | Class label (e.g. "cat") | Density map | Density map |
| **Columns** | 1 | 3 (parallel) | 1 |
| **Filter sizes** | Uniform (e.g. 3×3) | 9×9, 7×7, 5×5 (per column) | 3×3 (all layers) |
| **Pooling** | Yes (5×) | Yes (2× per column) | Yes (3×, then stopped) |
| **Output resolution** | 1/32 of input | 1/4 of input | 1/8 of input |
| **Receptive field** | Large (deep + pool) | Medium (shallow) | Very large (dilated convs) |
| **Pretrained** | Sometimes | No (custom filter sizes) | Yes (VGG16 frontend) |
| **Parameters** | ~14M (VGG) | ~130K (small) | ~16M (VGG + backend) |
| **Scale handling** | Single scale | 3 fixed scales | Continuous (dilation) |

---

## 5.2 Evolution of Ideas

### Step 1: Standard CNN → Fully Convolutional

```
Change:  Flatten + Dense → Conv(1×1)
Why:     Need spatial output (density map), not a single number
Result:  Can process any input size, output is a 2D map
```

### Step 2: Single Column → Multi-Column (MCNN)

```
Change:  One conv chain → Three parallel chains with different filter sizes
Why:     Heads at different scales need different receptive fields
Result:  Better scale handling, but expensive and no pretraining
```

### Step 3: Pooling → Dilated Convolution (CSRNet)

```
Change:  More pooling → Stop pooling + use dilated convs
Why:     Pooling loses spatial resolution needed for accurate density maps
Result:  High-resolution output + large receptive field + pretrained features
```

### The key trade-off visualized

```
MCNN approach:                    CSRNet approach:
                                  
3 shallow columns                 1 deep column
├── Large filters (9×9)          ├── VGG16 frontend (pretrained)
├── Medium filters (7×7)          ├── Dilated backend (large RF)
└── Small filters (5×5)          └── 1×1 conv output
    │                                 │
    └── Concat + 1×1 conv             └── High-res density map
        │
        └── Low-res density map
            (needs upsampling)
```

---

## 5.3 Quantitative Comparison (ShanghaiTech Part A)

| Model | MAE | MSE | Output size (512×512 input) |
|---|---|---|---|
| MCNN | 110.2 | 173.2 | 128×128 (1/4) |
| CSRNet | **68.2** | **115.0** | 64×64 (1/8) |

CSRNet reduces error by ~38% while having a smaller output map (but much richer features per pixel).

---

## 5.4 When to Use Which

### Use MCNN when:

- You have **lots of training data** (no pretraining available)
- The crowd has **extreme scale variation** (both very near and very far)
- You need a **small model** (~130K params vs ~16M for CSRNet)
- You're deploying on **resource-constrained devices** (edge, mobile)
- You want **interpretability** (each column clearly handles one scale)

### Use CSRNet when:

- You have **limited training data** (pretrained VGG helps a lot)
- You need **high accuracy** (CSRNet significantly outperforms MCNN)
- You want **sharp density maps** (less resolution loss)
- You have **GPU memory** for the larger model
- You're working with **highly congested scenes** (dilated convs see the whole image)

### Use neither when:

- You have a modern GPU and want the best results → use **ResNet-50/101 backbone** with dilated convs (CSRNet but with a better backbone)
- You need real-time video → use **lightweight models** (MobileNet-based)
- You have point annotations but no density maps → use **Bayesian loss** methods

---

## 5.5 What Came After CSRNet

| Year | Model | Key improvement over CSRNet |
|---|---|---|
| 2019 | **SANet** | Scale aggregation modules, attention |
| 2019 | **CAN** | Context-aware network, spatial attention |
| 2020 | **BL** | Bayesian loss — handles point annotations directly |
| 2020 | **DM-Count** | Distribution matching, optimal transport |
| 2021 | **ASNet** | Adaptive scale network |
| 2022 | **CLIP-based** | Zero-shot crowd counting |

But CSRNet remains the **baseline** — it's simple, effective, and every new method compares against it.

---

## 5.6 Summary: The Mental Model

```
CNN:        "What is this?" → single label
              │
              ▼
FCN:        "Where are things and how many?" → density map
              │
       ┌──────┴──────┐
       ▼              ▼
     MCNN           CSRNet
  "Different       "Keep resolution,
   scales need       use pretrained
   different         features, grow
   filters"          RF with dilation"
       │              │
       └──────┬───────┘
              ▼
     Modern crowd counting
     (attention, transformers,
      Bayesian methods)
```

**The core tension in all of this:** receptive field vs. resolution. You want the network to see the whole image (large RF) but also output a sharp density map (high resolution). Pooling gives you RF but kills resolution. Dilated convolution gives you both. That's the single most important architectural insight in this space.
