from __future__ import annotations

import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "stokes_metric_plot_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _plot_series(ax, epochs, series, label, *, color=None, linestyle="-"):
    values = np.asarray(series, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return
    ax.plot(epochs, values, label=label, color=color, linestyle=linestyle, linewidth=2.0)


def _finalize_axis(ax, title, ylabel):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False, fontsize=9)


def save_training_metrics_figure(
    history,
    *,
    save_path,
    gen_metric_label,
    op_metric_label,
    config_lines,
):
    epochs = np.asarray(history["epoch"], dtype=float)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    axes = axes.reshape(-1)

    _plot_series(axes[0], epochs, history["gen_loss"], gen_metric_label, color="#0f766e")
    _finalize_axis(axes[0], "Generator", gen_metric_label)

    _plot_series(axes[1], epochs, history["op_loss"], op_metric_label, color="#1d4ed8")
    _plot_series(axes[1], epochs, history["op_rel_loss"], "op_train_rel_l2", color="#9333ea")
    _finalize_axis(axes[1], "Operator", "Loss")

    _plot_series(axes[2], epochs, history["op_div_loss"], "op_train_pred_div_meanabs", color="#dc2626")
    _finalize_axis(axes[2], "Operator Divergence", "Mean |div|")

    _plot_series(axes[3], epochs, history["heldout_mean_rel_l2"], "heldout_mean_rel_l2", color="#2563eb")
    _plot_series(axes[3], epochs, history["heldout_worst_rel_l2"], "heldout_worst_rel_l2", color="#ea580c")
    _plot_series(axes[3], epochs, history["heldout_worst_rel_l2_best"], "heldout_worst_rel_l2_best", color="#059669")
    _finalize_axis(axes[3], "Held-Out Rel L2", "Relative Error")

    _plot_series(axes[4], epochs, history["heldout_mean_pred_div_meanabs"], "heldout_mean_pred_div_meanabs", color="#2563eb")
    _plot_series(axes[4], epochs, history["heldout_worst_pred_div_meanabs"], "heldout_worst_pred_div_meanabs", color="#ea580c")
    _plot_series(
        axes[4],
        epochs,
        history["heldout_worst_pred_div_meanabs_best"],
        "heldout_worst_pred_div_meanabs_best",
        color="#059669",
    )
    _finalize_axis(axes[4], "Held-Out Predicted Divergence", "Mean |div|")

    _plot_series(axes[5], epochs, history["sig_mean"], "sig_mean", color="#7c3aed")
    _finalize_axis(axes[5], "Sampler Scale", "Mean sigma")

    axes[6].axis("off")
    axes[6].text(
        0.0,
        1.0,
        "\n".join(config_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )
    axes[6].set_title("Run Config")

    axes[7].axis("off")

    fig.suptitle("Stokes KNO Training Metrics", fontsize=18)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path.resolve()
