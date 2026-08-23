#!/usr/bin/env python3
"""Plot training curves for the MCNN vs CSRNet report figure.

Generates `reports/figures/training_curves.png` with two panels:

1. Validation MAE vs epoch for both models (log y-axis), showing MCNN's steady
   descent to a plateau and CSRNet's much earlier best (epoch 34) followed by
   rising val MAE.
2. CSRNet train vs validation MAE, showing the overfitting divergence after
   epoch 34 (train keeps dropping, val rises).

The per-epoch values below are transcribed verbatim from the training runs
preserved in `notes/19-mcnn-learning-rate-tuning.md` (MCNN lr=1e-5 run) and the
CSRNet Colab run (full log archived alongside this figure). They are kept
inline so the figure is reproducible without re-running training.

Run:
    uv run --locked python scripts/plot_training_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures" / "training_curves.png"

# --- MCNN (lr=1e-5, Part A) — from notes/19 §19.10 ---------------------------
mcnn_epochs = list(range(1, 51))
mcnn_train = [
    4341.32, 3923.39, 3512.96, 3138.69, 2807.51, 2509.59, 2247.04, 2014.27,
    1809.31, 1628.53, 1467.57, 1324.65, 1200.51, 1089.99, 990.86, 913.34,
    836.61, 771.91, 717.29, 672.26, 633.33, 597.37, 562.76, 544.51, 525.15,
    508.80, 493.14, 482.45, 471.33, 463.83, 454.57, 445.66, 440.41, 432.73,
    430.41, 421.89, 421.28, 416.16, 412.70, 408.92, 406.17, 403.53, 397.18,
    397.36, 397.43, 395.81, 391.91, 390.78, 387.42, 385.94,
]
mcnn_val = [
    4133.11, 3701.73, 3319.75, 2971.54, 2663.75, 2392.43, 2147.17, 1931.72,
    1744.70, 1576.90, 1427.15, 1297.34, 1183.91, 1081.05, 992.40, 914.47,
    841.53, 781.76, 727.91, 683.98, 644.60, 617.59, 594.00, 575.87, 558.14,
    543.87, 533.88, 524.22, 515.35, 508.09, 502.10, 496.84, 492.13, 487.78,
    484.05, 481.87, 480.35, 478.46, 476.95, 475.72, 474.56, 473.39, 472.19,
    471.06, 469.89, 468.74, 467.59, 466.70, 465.79, 464.74,
]

# --- CSRNet (lr=1e-5, Part A) — from the Colab run --------------------------
csrnet_epochs = list(range(1, 51))
csrnet_train = [
    512.61, 499.91, 484.94, 469.43, 458.76, 447.59, 436.01, 424.96, 414.88,
    405.78, 396.52, 382.80, 372.56, 366.83, 360.59, 352.48, 347.81, 343.81,
    338.62, 333.01, 325.53, 322.35, 319.42, 315.24, 312.06, 307.97, 304.59,
    302.50, 296.82, 296.26, 297.04, 294.89, 293.11, 291.72, 284.87, 287.38,
    287.20, 286.56, 285.08, 281.99, 284.30, 282.27, 277.90, 282.51, 277.90,
    280.70, 279.77, 280.11, 279.73, 278.71,
]
csrnet_val = [
    543.77, 530.21, 516.96, 504.44, 492.14, 479.73, 467.86, 456.22, 445.15,
    434.59, 425.44, 416.27, 407.81, 400.81, 393.83, 386.93, 380.23, 374.50,
    368.54, 363.01, 358.20, 353.43, 349.20, 345.19, 341.32, 338.32, 335.82,
    333.97, 332.22, 331.29, 330.45, 329.61, 328.87, 328.22, 328.94, 329.63,
    330.39, 331.50, 332.71, 333.98, 335.16, 336.47, 337.61, 338.79, 339.91,
    340.91, 342.09, 343.08, 344.08, 345.15,
]

CSRNET_BEST_EPOCH = 34   # min val MAE = 328.22
MCNN_BEST_EPOCH = 50     # min val MAE = 464.74 (still descending)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Panel 1: validation MAE, both models (log y) -----------------------
    ax1.plot(mcnn_epochs, mcnn_val, "o-", color="#1f77b4", ms=3, lw=1.4,
             label="MCNN (val)")
    ax1.plot(csrnet_epochs, csrnet_val, "s-", color="#d62728", ms=3, lw=1.4,
             label="CSRNet (val)")
    ax1.scatter([MCNN_BEST_EPOCH], [min(mcnn_val)], color="#1f77b4",
                zorder=5, s=90, edgecolor="black", linewidth=0.8)
    ax1.scatter([CSRNET_BEST_EPOCH], [min(csrnet_val)], color="#d62728",
                zorder=5, s=90, marker="s", edgecolor="black", linewidth=0.8)
    ax1.annotate(f"best ep {CSRNET_BEST_EPOCH}\nval {min(csrnet_val):.0f}",
                 xy=(CSRNET_BEST_EPOCH, min(csrnet_val)),
                 xytext=(CSRNET_BEST_EPOCH + 3, min(csrnet_val) + 140),
                 fontsize=9, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation MAE (log scale)")
    ax1.set_title("Validation MAE — MCNN plateaus, CSRNet overfits then rises")
    ax1.legend(loc="upper right")
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.set_xlim(0, 52)

    # --- Panel 2: CSRNet train vs val (overfitting) ------------------------
    ax2.plot(csrnet_epochs, csrnet_train, "o-", color="#2ca02c", ms=3, lw=1.4,
             label="CSRNet train")
    ax2.plot(csrnet_epochs, csrnet_val, "s-", color="#d62728", ms=3, lw=1.4,
             label="CSRNet val")
    ax2.axvline(CSRNET_BEST_EPOCH, color="gray", ls="--", lw=1,
                label=f"best epoch {CSRNET_BEST_EPOCH}")
    # shade the overfitting region (val rising while train falls)
    ax2.axvspan(CSRNET_BEST_EPOCH, 50, color="red", alpha=0.06)
    ax2.text(42, 470, "overfitting:\ntrain↓  val↑", fontsize=9, color="red",
             ha="center")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.set_title("CSRNet — train vs validation (capacity vs data-size trade-off)")
    ax2.legend(loc="upper right")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.set_xlim(0, 52)

    fig.suptitle(
        "Training dynamics: MCNN vs CSRNet on ShanghaiTech Part A (50 epochs)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()