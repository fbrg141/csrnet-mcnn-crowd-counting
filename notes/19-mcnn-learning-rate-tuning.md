# 19 — MCNN Learning Rate Tuning (Part A)

Findings from the first real training run on Colab (T4 GPU), used to justify the
lr choice in the final comparison. This is the "negative result" row of the
results table, plus the rationale for the corrected run.

---

## 19.1 Setup

- Model: MCNN, from scratch, `src/models/mcnn.py` (params = 64,385)
- Data: ShanghaiTech Part A, 270 train / 30 val (`VAL_SPLIT = 0.1`)
- Loss: pixel-wise MSE (Euclidean loss, paper convention)
- Optimizer: SGD, `momentum = 0.95`
- lr = 1e-6, epochs = 50, batch = 4 → **3,375 iterations total**
- Counting: `density.sum()` for both prediction and GT (issue: counting convention)
- seed = 42

```
uv run --locked python -m src.train --model mcnn --part A --lr 1e-6 --epochs 50
```

## 19.2 Observation: linear, non-converging descent

| epoch | loss    | train_mae | val_mae | Δ val_mae/epoch |
|------:|--------:|----------:|--------:|----------------:|
| 1     | 0.00907 | 4465.76   | 4434.06 | —               |
| 5     | 0.00839 | 4285.85   | 4248.48 | ~−46            |
| 10    | 0.00757 | 4059.24   | 4027.47 | ~−44            |
| 17    | 0.00655 | 3758.36   | 3737.06 | ~−41            |
| 50    | 0.00350 | 2627.71   | 2629.85 | ~−34 (avg 17→50) |

- The MAE decrease is **linear**, not exponential/flattening — the model is
  nowhere near a minimum; it is simply sliding down a near-straight slope.
- Initial val MAE ≈ 4400 is the from-scratch init: the network outputs a density
  map summing to ~4700 before learning (CSRNet does not have this problem
  because its VGG16 frontend is pretrained).
- Final (epoch 50): **val_mae = 2629.85** (train_mae 2627.71, loss 0.00350).
  The slope decelerated slightly (early ~−45/epoch → late ~−34/epoch) but
  never flattened — still linear, still descending at run end. Extrapolating
  linearly, reaching the paper's 110 would need ~70 more epochs (~3600 more
  iterations), and linear extrapolation breaks near convergence anyway.
- Paper MCNN Part A: **MAE ≈ 110**.

So this run, as configured, did not produce a number that reflects MCNN's
capability — it is severely undertrained. Confirmed by the linear, never-
flattening loss/MAE curve.

## 19.3 Root cause: lr / iteration-budget mismatch

The README pins `lr = 1e-6` to the paper (Zhang et al., CVPR 2016). That is
faithful to the paper's *lr*, but not to the paper's *training budget*:

|              | paper (approx) | this run |
|--------------|----------------:|---------:|
| lr           | 1e-6            | 1e-6     |
| batch        | 1               | 4        |
| iterations   | ~100,000+       | ~3,375   |
| epochs       | hundreds        | 50       |

A from-scratch network initialized to output ~4700-head density maps cannot be
pulled down to ~100-head predictions in 3.4k steps at lr=1e-6. Faithfulness has
to match **both** lr and the iteration count, or neither.

## 19.4 Decision

Two honest options:

1. Keep `lr = 1e-6` and raise epochs to ~300 (Colab free-tier time limits make
   this impractical, and a single session may be disconnected before it ends).
2. Raise lr to 1e-5 (or 1e-4) and keep 50 epochs — deviates from the paper lr,
   but matches the paper's *effective gradient budget* far better.

**Chosen: option 2**, with the deviation documented here and in the report.
The lr=1e-6 run is kept as a baseline / negative-result row to demonstrate the
training-dynamics analysis, not as MCNN's reported performance.

## 19.5 Results table (filled in after the runs)

| Run                   | lr    | epochs | val MAE | note                         |
|-----------------------|------:|-------:|--------:|------------------------------|
| paper-faithful lr     | 1e-6  | 50     | 2629.85 | undertrained (section 19.2)  |
| matched budget        | 1e-5  | 50     | 464.74  | converged (section 19.9)     |
| matched budget (lr1e-4) | 1e-4 | 50     | TBD     | only if 1e-5 also undertrains |

## 19.6 Commands used

```bash
# baseline (undertrained, for the table's negative row)
uv run --locked python -m src.train --model mcnn --part A --lr 1e-6 --epochs 50 \
    --out-dir reports/checkpoints/lr1e-6

# corrected run (real number)
uv run --locked python -m src.train --model mcnn --part A --lr 1e-5 --epochs 50 \
    --out-dir reports/checkpoints/lr1e-5

# if 1e-5 also undertrains:
uv run --locked python -m src.train --model mcnn --part A --lr 1e-4 --epochs 50 \
    --out-dir reports/checkpoints/lr1e-4
```

Separate `--out-dir` folders keep the three checkpoints distinct; each folder
still writes `mcnn_partA_best.pth` inside it, so the `--out-dir` is what
separates the runs.

## 19.7 Triage rule (for picking lr without babysitting)

Run 5 epochs at each of 1e-6, 1e-5, 1e-4 and read the slope:

| lr   | expected behavior                          |
|------|--------------------------------------------|
| 1e-6 | slow steady drop (~−45 MAE/epoch)           |
| 1e-5 | ~10× faster drop — good candidate           |
| 1e-4 | fast drop **or** loss → nan (too big)       |
| 1e-3 | almost certainly diverges                   |

Pick the largest lr that descends smoothly without `loss → nan`.

## 19.9 lr=1e-5 run result (converged)

Final: **val_mae = 464.74** (train_mae 385.94, val_rmse 825.97, loss 0.000815).

Descent profile:

| epoch | val_mae | phase |
|------:|--------:|------|
| 1     | 4133    | init (~4700-head density output) |
| 20    | 684     | fast descent complete |
| 30    | 508     | plateau forming |
| 50    | 465     | flat (epoch 40→50: ~−1/epoch) |

Unlike the lr=1e-6 run, this curve shows a clear **plateau** from epoch ~30:
epoch 30→50 only dropped 508→465 (~−2/epoch), so the model reached its capacity
at this fixed lr / epoch budget. The val/train gap (465 vs 386) is mild — no
severe overfitting.

**Comparison to the paper:** MCNN reports MAE ≈ 110 on Part A. We reach 464.74.
The remaining ~350-point gap is attributable to the documented deviations that
are still outstanding (not lr): no data augmentation (#14), no lr scheduling,
fixed-sigma (not adaptive) density maps, and fewer epochs than the paper. The
lr change alone took MAE from 2630 → 465 — the dominant lever in this run.

Note: 464.74 is **validation** MAE (30 held-out train images). The reported
comparison number must be **test** MAE over the 182 Part A test images, obtained
via `src.evaluate` (see section 19.10).

### Test-set result

```
[device] cuda
[model] mcnn params=64385 (ckpt epoch=50 val_mae=464.73787078857424)
[data] test=182 images
[result] MAE=315.2272  RMSE=429.1577
[saved] reports/mcnn_partA_seed42_metrics.json
```

| metric        | value  |
|---------------|-------:|
| test MAE      | 315.23 |
| test RMSE     | 429.16 |
| val MAE       | 464.74 |
| params        | 64,385 |
| paper MAE     | ~110   |
| paper RMSE    | ~173   |

Test MAE (315.23) is **lower** than val MAE (464.74) — the 30-image val set is
noisier than the 182-image test set; the test number is the one to report. The
~200-point gap to the paper remains attributable to the still-outstanding
deviations (no augmentation #14, no lr scheduling, fixed-sigma density maps,
fewer epochs).

## 19.8 Appendix: full epoch log (lr=1e-6 run)

Raw output of the baseline run, preserved verbatim for the record. The slope
is visibly near-constant throughout (early ~−45/epoch, late ~−34/epoch), with
no plateau — the signature of a non-converging, undertrained run.

```
[device] cuda
[config] model=mcnn part=A downsample=4 normalize=False lr=1e-06 momentum=0.95 seed=42
[model] mcnn params=64385
[data] train=270 val=30
[epoch   1] loss=0.009070 train_mae=4465.76 train_rmse=4499.09 | val_mae=4434.06 val_rmse=4481.28
[epoch   2] loss=0.008909 train_mae=4424.11 train_rmse=4458.14 | val_mae=4386.79 val_rmse=4434.50
[epoch   3] loss=0.008747 train_mae=4381.71 train_rmse=4415.10 | val_mae=4340.63 val_rmse=4388.83
[epoch   4] loss=0.008558 train_mae=4332.04 train_rmse=4365.76 | val_mae=4294.21 val_rmse=4342.92
[epoch   5] loss=0.008386 train_mae=4285.85 train_rmse=4319.37 | val_mae=4248.48 val_rmse=4297.72
[epoch   6] loss=0.008209 train_mae=4238.51 train_rmse=4271.89 | val_mae=4203.61 val_rmse=4253.38
[epoch   7] loss=0.008047 train_mae=4193.65 train_rmse=4227.25 | val_mae=4158.52 val_rmse=4208.84
[epoch   8] loss=0.007888 train_mae=4149.48 train_rmse=4182.96 | val_mae=4114.27 val_rmse=4165.15
[epoch   9] loss=0.007728 train_mae=4105.25 train_rmse=4138.45 | val_mae=4070.83 val_rmse=4122.28
[epoch  10] loss=0.007566 train_mae=4059.24 train_rmse=4092.55 | val_mae=4027.47 val_rmse=4079.51
[epoch  11] loss=0.007404 train_mae=4012.94 train_rmse=4046.18 | val_mae=3984.20 val_rmse=4036.84
[epoch  12] loss=0.007242 train_mae=3966.03 train_rmse=3998.96 | val_mae=3942.07 val_rmse=3995.32
[epoch  13] loss=0.007100 train_mae=3925.34 train_rmse=3958.52 | val_mae=3900.36 val_rmse=3954.24
[epoch  14] loss=0.006973 train_mae=3886.34 train_rmse=3919.71 | val_mae=3858.99 val_rmse=3913.51
[epoch  15] loss=0.006825 train_mae=3842.20 train_rmse=3875.49 | val_mae=3818.07 val_rmse=3873.24
[epoch  16] loss=0.006680 train_mae=3798.99 train_rmse=3832.15 | val_mae=3777.35 val_rmse=3833.20
[epoch  17] loss=0.006546 train_mae=3758.36 train_rmse=3791.95 | val_mae=3737.06 val_rmse=3793.59
[epoch  18] loss=0.006435 train_mae=3722.05 train_rmse=3755.53 | val_mae=3697.41 val_rmse=3754.63
[epoch  19] loss=0.006291 train_mae=3677.36 train_rmse=3710.86 | val_mae=3658.13 val_rmse=3716.06
[epoch  20] loss=0.006175 train_mae=3640.61 train_rmse=3674.41 | val_mae=3619.17 val_rmse=3677.82
[epoch  21] loss=0.006060 train_mae=3602.64 train_rmse=3636.56 | val_mae=3580.91 val_rmse=3640.30
[epoch  22] loss=0.005941 train_mae=3563.46 train_rmse=3597.27 | val_mae=3542.85 val_rmse=3602.99
[epoch  23] loss=0.005757 train_mae=3516.56 train_rmse=3549.30 | val_mae=3505.21 val_rmse=3566.11
[epoch  24] loss=0.005698 train_mae=3484.16 train_rmse=3517.96 | val_mae=3467.73 val_rmse=3529.41
[epoch  25] loss=0.005600 train_mae=3450.09 train_rmse=3484.24 | val_mae=3431.14 val_rmse=3493.61
[epoch  26] loss=0.005486 train_mae=3411.60 train_rmse=3445.95 | val_mae=3394.47 val_rmse=3457.75
[epoch  27] loss=0.005387 train_mae=3376.47 train_rmse=3410.82 | val_mae=3358.29 val_rmse=3422.39
[epoch  28] loss=0.005273 train_mae=3337.42 train_rmse=3371.95 | val_mae=3322.73 val_rmse=3387.66
[epoch  29] loss=0.005176 train_mae=3302.40 train_rmse=3337.15 | val_mae=3287.38 val_rmse=3353.15
[epoch  30] loss=0.005068 train_mae=3264.29 train_rmse=3299.28 | val_mae=3252.39 val_rmse=3319.03
[epoch  31] loss=0.004982 train_mae=3232.36 train_rmse=3267.44 | val_mae=3217.72 val_rmse=3285.24
[epoch  32] loss=0.004882 train_mae=3195.47 train_rmse=3230.79 | val_mae=3183.60 val_rmse=3252.01
[epoch  33] loss=0.004777 train_mae=3158.21 train_rmse=3193.59 | val_mae=3149.86 val_rmse=3219.17
[epoch  34] loss=0.004690 train_mae=3125.50 train_rmse=3161.30 | val_mae=3116.43 val_rmse=3186.66
[epoch  35] loss=0.004611 train_mae=3093.87 train_rmse=3129.82 | val_mae=3083.26 val_rmse=3154.42
[epoch  36] loss=0.004523 train_mae=3059.68 train_rmse=3095.87 | val_mae=3050.46 val_rmse=3122.58
[epoch  37] loss=0.004443 train_mae=3028.25 train_rmse=3064.37 | val_mae=3018.39 val_rmse=3091.46
[epoch  38] loss=0.004358 train_mae=2994.68 train_rmse=3031.39 | val_mae=2986.23 val_rmse=3060.28
[epoch  39] loss=0.004275 train_mae=2961.35 train_rmse=2998.38 | val_mae=2954.57 val_rmse=3029.61
[epoch  40] loss=0.004197 train_mae=2930.08 train_rmse=2966.97 | val_mae=2923.50 val_rmse=2999.54
[epoch  41] loss=0.004122 train_mae=2899.26 train_rmse=2936.54 | val_mae=2892.50 val_rmse=2969.57
[epoch  42] loss=0.004045 train_mae=2867.42 train_rmse=2904.97 | val_mae=2862.04 val_rmse=2940.14
[epoch  43] loss=0.003949 train_mae=2831.10 train_rmse=2868.64 | val_mae=2831.83 val_rmse=2910.98
[epoch  44] loss=0.003895 train_mae=2805.17 train_rmse=2843.41 | val_mae=2801.90 val_rmse=2882.11
[epoch  45] loss=0.003829 train_mae=2775.48 train_rmse=2813.99 | val_mae=2772.45 val_rmse=2853.73
[epoch  46] loss=0.003759 train_mae=2745.60 train_rmse=2784.45 | val_mae=2743.23 val_rmse=2825.60
[epoch  47] loss=0.003690 train_mae=2715.48 train_rmse=2754.61 | val_mae=2714.46 val_rmse=2797.94
[epoch  48] loss=0.003628 train_mae=2687.21 train_rmse=2726.71 | val_mae=2685.70 val_rmse=2770.30
[epoch  49] loss=0.003566 train_mae=2659.11 train_rmse=2698.83 | val_mae=2657.68 val_rmse=2743.41
[epoch  50] loss=0.003497 train_mae=2627.71 train_rmse=2667.76 | val_mae=2629.85 val_rmse=2716.73
[done] best val_mae=2629.85 -> reports/checkpoints/mcnn_partA_seed42_best.pth
```

Checkpoint path on Colab: `reports/checkpoints/mcnn_partA_seed42_best.pth`
(ephemeral — copy to Drive before disconnect, see README Colab notes).

## 19.10 lr=1e-5 full epoch log

```
[device] cuda
[config] model=mcnn part=A downsample=4 normalize=False lr=1e-05 momentum=0.95 seed=42
[model] mcnn params=64385
[data] train=270 val=30
[epoch   1] loss=0.008597 train_mae=4341.32 train_rmse=4375.18 | val_mae=4133.11 val_rmse=4183.74
[epoch   2] loss=0.007109 train_mae=3923.39 train_rmse=3960.64 | val_mae=3701.73 val_rmse=3758.88
[epoch   3] loss=0.005789 train_mae=3512.96 train_rmse=3547.08 | val_mae=3319.75 val_rmse=3384.74
[epoch   4] loss=0.004735 train_mae=3138.69 train_rmse=3176.54 | val_mae=2971.54 val_rmse=3046.06
[epoch   5] loss=0.003905 train_mae=2807.51 train_rmse=2846.15 | val_mae=2663.75 val_rmse=2749.25
[epoch   6] loss=0.003247 train_mae=2509.59 train_rmse=2551.57 | val_mae=2392.43 val_rmse=2490.30
[epoch   7] loss=0.002742 train_mae=2247.04 train_rmse=2295.73 | val_mae=2147.17 val_rmse=2259.17
[epoch   8] loss=0.002347 train_mae=2014.27 train_rmse=2069.16 | val_mae=1931.72 val_rmse=2059.16
[epoch   9] loss=0.002034 train_mae=1809.31 train_rmse=1868.32 | val_mae=1744.70 val_rmse=1888.55
[epoch  10] loss=0.001792 train_mae=1628.53 train_rmse=1694.24 | val_mae=1576.90 val_rmse=1738.58
[epoch  11] loss=0.001603 train_mae=1467.57 train_rmse=1542.18 | val_mae=1427.15 val_rmse=1607.87
[epoch  12] loss=0.001455 train_mae=1324.65 train_rmse=1407.24 | val_mae=1297.34 val_rmse=1497.53
[epoch  13] loss=0.001336 train_mae=1200.51 train_rmse=1293.21 | val_mae=1183.91 val_rmse=1403.90
[epoch  14] loss=0.001252 train_mae=1089.99 train_rmse=1193.47 | val_mae=1081.05 val_rmse=1321.69
[epoch  15] loss=0.001179 train_mae=990.86 train_rmse=1104.47 | val_mae=992.40 val_rmse=1253.27
[epoch  16] loss=0.001125 train_mae=913.34 train_rmse=1036.11 | val_mae=914.47 val_rmse=1195.29
[epoch  17] loss=0.001079 train_mae=836.61 train_rmse=972.48 | val_mae=841.53 val_rmse=1143.15
[epoch  18] loss=0.001051 train_mae=771.91 train_rmse=916.15 | val_mae=781.76 val_rmse=1102.04
[epoch  19] loss=0.001020 train_mae=717.29 train_rmse=869.46 | val_mae=727.91 val_rmse=1066.43
[epoch  20] loss=0.000996 train_mae=672.26 train_rmse=830.02 | val_mae=683.98 val_rmse=1036.62
[epoch  21] loss=0.000983 train_mae=633.33 train_rmse=795.91 | val_mae=644.60 val_rmse=1008.98
[epoch  22] loss=0.000967 train_mae=597.37 train_rmse=764.30 | val_mae=617.59 val_rmse=988.31
[epoch  23] loss=0.000912 train_mae=562.76 train_rmse=731.38 | val_mae=594.00 val_rmse=969.38
[epoch  24] loss=0.000939 train_mae=544.51 train_rmse=716.71 | val_mae=575.87 val_rmse=954.32
[epoch  25] loss=0.000933 train_mae=525.15 train_rmse=699.68 | val_mae=558.14 val_rmse=939.68
[epoch  26] loss=0.000924 train_mae=508.80 train_rmse=683.50 | val_mae=543.87 val_rmse=927.18
[epoch  27] loss=0.000918 train_mae=493.14 train_rmse=668.37 | val_mae=533.88 val_rmse=917.40
[epoch  28] loss=0.000910 train_mae=482.45 train_rmse=656.79 | val_mae=524.22 val_rmse=908.21
[epoch  29] loss=0.000903 train_mae=471.33 train_rmse=645.86 | val_mae=515.35 val_rmse=899.60
[epoch  30] loss=0.000897 train_mae=463.83 train_rmse=636.71 | val_mae=508.09 val_rmse=892.16
[epoch  31] loss=0.000893 train_mae=454.57 train_rmse=627.50 | val_mae=502.10 val_rmse=885.81
[epoch  32] loss=0.000888 train_mae=445.66 train_rmse=619.11 | val_mae=496.84 val_rmse=880.07
[epoch  33] loss=0.000878 train_mae=440.41 train_rmse=613.26 | val_mae=492.13 val_rmse=875.06
[epoch  34] loss=0.000872 train_mae=432.73 train_rmse=606.22 | val_mae=487.78 val_rmse=870.56
[epoch  35] loss=0.000873 train_mae=430.41 train_rmse=602.55 | val_mae=484.05 val_rmse=865.68
[epoch  36] loss=0.000868 train_mae=421.89 train_rmse=594.95 | val_mae=481.87 val_rmse=861.57
[epoch  37] loss=0.000866 train_mae=421.28 train_rmse=593.47 | val_mae=480.35 val_rmse=859.13
[epoch  38] loss=0.000860 train_mae=416.16 train_rmse=588.64 | val_mae=478.46 val_rmse=854.43
[epoch  39] loss=0.000857 train_mae=412.70 train_rmse=584.78 | val_mae=476.95 val_rmse=851.31
[epoch  40] loss=0.000853 train_mae=408.92 train_rmse=581.59 | val_mae=475.72 val_rmse=849.46
[epoch  41] loss=0.000849 train_mae=406.17 train_rmse=578.31 | val_mae=474.56 val_rmse=846.17
[epoch  42] loss=0.000844 train_mae=403.53 train_rmse=574.78 | val_mae=473.39 val_rmse=843.52
[epoch  43] loss=0.000831 train_mae=397.18 train_rmse=568.86 | val_mae=472.19 val_rmse=841.31
[epoch  44] loss=0.000834 train_mae=397.36 train_rmse=568.90 | val_mae=471.06 val_rmse=838.52
[epoch  45] loss=0.000833 train_mae=397.43 train_rmse=567.87 | val_mae=469.89 val_rmse=836.23
[epoch  46] loss=0.000828 train_mae=395.81 train_rmse=565.38 | val_mae=468.74 val_rmse=833.68
[epoch  47] loss=0.000824 train_mae=391.91 train_rmse=562.67 | val_mae=467.59 val_rmse=831.84
[epoch  48] loss=0.000822 train_mae=390.78 train_rmse=560.74 | val_mae=466.70 val_rmse=829.48
[epoch  49] loss=0.000819 train_mae=387.42 train_rmse=557.01 | val_mae=465.79 val_rmse=827.38
[epoch  50] loss=0.000815 train_mae=385.94 train_rmse=556.11 | val_mae=464.74 val_rmse=825.97
[done] best val_mae=464.74 -> reports/checkpoints/mcnn_partA_seed42_best.pth
```