from __future__ import annotations

from pathlib import Path

WANDB_API_KEY = "d612cda26a5690e196d092756d668fc2aee8525b"


def init_wandb_run(*, project, name, config, static_metrics):
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required for --wandb") from exc

    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=project, name=name or None)
    wandb.config.update(config)
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    if static_metrics:
        wandb.log(static_metrics, step=0)
    return wandb


def log_wandb_metrics(wandb_module, metrics, *, step):
    if wandb_module is None:
        return
    wandb_module.log(metrics, step=step)


def finish_wandb_run(wandb_module, *, metrics_plot_path):
    if wandb_module is None:
        return

    metrics_plot_path = Path(metrics_plot_path).resolve()
    wandb_module.run.summary["saved_metrics_plot"] = str(metrics_plot_path)
    if metrics_plot_path.exists():
        wandb_module.log({"metrics_plot": wandb_module.Image(str(metrics_plot_path))})
    wandb_module.finish()
