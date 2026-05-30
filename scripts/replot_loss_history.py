"""Post-processing script to replot pinn_loss_history.png from loss_history.csv with EMA smoothing."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "loss_history.csv"
OUT_PATH = ROOT / "graphs" / "pinn_loss_history.png"

def ema(values: np.ndarray, alpha: float = 0.04) -> np.ndarray:
    if len(values) == 0:
        return values
    smoothed = []
    current = values[0]
    for val in values:
        if np.isnan(val):
            smoothed.append(np.nan)
        else:
            if np.isnan(current):
                current = val
            current = alpha * val + (1.0 - alpha) * current
            smoothed.append(current)
    return np.asarray(smoothed)

def main() -> None:
    if not CSV_PATH.is_file():
        print(f"Error: {CSV_PATH} not found.")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} epochs from {CSV_PATH.name}")
    
    epochs = df["epoch"].astype(float).values
    data_loss = df["data_mse"].astype(float).values
    physics_loss = df["physics_mse"].astype(float).values
    
    # Determine pretrain boundary
    pre_mask = df["phase"] == "pretrain"
    if pre_mask.any():
        train_pre = int(df[pre_mask]["epoch"].max())
    else:
        train_pre = 600
        
    pre_m = epochs <= train_pre
    joint_m = epochs > train_pre

    fig, (ax_pre, ax_joint) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Physics Pre-training
    if np.any(pre_m):
        epochs_pre = epochs[pre_m]
        phys_pre = np.clip(physics_loss[pre_m], 1e-30, None)
        phys_pre_smooth = ema(phys_pre, alpha=0.08)
        
        # Raw loss in light color
        ax_pre.semilogy(epochs_pre, phys_pre, color="#e74c3c", alpha=0.15, label="Raw Physics MSE")
        # Smoothed loss in bold
        ax_pre.semilogy(epochs_pre, phys_pre_smooth, color="#e74c3c", lw=2, label="Smoothed Physics MSE")
        
        ax_pre.set_title(f"Phase 1: Physics Pre-training (epochs 1–{train_pre})", fontsize=12, fontweight="bold", pad=10)
        ax_pre.set_xlabel("Epoch")
        ax_pre.set_ylabel("Loss (log scale)")
        ax_pre.legend(loc="upper right")
        ax_pre.grid(True, which="both", ls="-", alpha=0.25)
    else:
        ax_pre.text(0.5, 0.5, "No pretrain epochs logged", ha="center", va="center", transform=ax_pre.transAxes)
        ax_pre.set_axis_off()

    # Panel 2: Joint Training
    if np.any(joint_m):
        epochs_joint = epochs[joint_m]
        data_joint = np.where(data_loss[joint_m] > 0.0, data_loss[joint_m], np.nan)
        phys_joint = np.clip(physics_loss[joint_m], 1e-30, None)
        
        data_joint_smooth = ema(data_joint, alpha=0.04)
        phys_joint_smooth = ema(phys_joint, alpha=0.04)
        
        # Raw losses
        ax_joint.semilogy(epochs_joint, data_joint, color="#3498db", alpha=0.15, label="Raw Data MSE")
        ax_joint.semilogy(epochs_joint, phys_joint, color="#e74c3c", alpha=0.15, label="Raw Physics MSE")
        
        # Smoothed losses
        ax_joint.semilogy(epochs_joint, data_joint_smooth, color="#3498db", lw=2, label="Smoothed Data MSE")
        ax_joint.semilogy(epochs_joint, phys_joint_smooth, color="#e74c3c", lw=2, label="Smoothed Physics MSE")
        
        ax_joint.set_title("Phase 2: Joint Training", fontsize=12, fontweight="bold", pad=10)
        ax_joint.set_xlabel("Epoch")
        ax_joint.legend(loc="upper right")
        ax_joint.grid(True, which="both", ls="-", alpha=0.25)
    else:
        ax_joint.text(0.5, 0.5, "No joint epochs logged", ha="center", va="center", transform=ax_joint.transAxes)
        ax_joint.set_axis_off()

    fig.suptitle("Physics-Informed Neural Network (PINN) Loss Convergence", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)
    print(f"Saved smoothed loss plot to {OUT_PATH.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
