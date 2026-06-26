import os
import logging
import torch
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(
    log_dir: str,
    name: str = "flik",
    level: int = logging.INFO,
    tensorboard: bool = True,
    wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_offline: bool = False,
    config: Optional[Dict[str, Any]] = None,
    rank: int = 0,
) -> Dict[str, Any]:
    """
    Setup logging (console + file) and optional TensorBoard/WandB.
    When rank > 0 (non-zero DDP ranks), the file handler is skipped,
    the console handler is set to WARNING, and TB/WandB are disabled.
    Returns dictionary with loggers and writers.
    """
    os.makedirs(log_dir, exist_ok=True)

    # File handler (only rank 0 writes to file)
    if rank == 0:
        log_file = os.path.join(
            log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level if rank == 0 else logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if rank == 0:
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if rank == 0:
        root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Module logger
    logger = logging.getLogger(name)

    # TensorBoard SummaryWriter (rank 0 only)
    tb_writer = None
    if tensorboard and rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard logging to {tb_dir}")

    # WandB run (rank 0 only)
    wandb_run = None
    if wandb and rank == 0:
        import wandb as wandb_lib

        wandb_run = wandb_lib.init(
            project=wandb_project or "flik-video-grounding",
            entity=wandb_entity,
            config=config,
            dir=log_dir,
            mode="offline" if wandb_offline else None,
        )
        logger.info(f"WandB run: {wandb_run.name}")

    return {
        "logger": logger,
        "tb_writer": tb_writer,
        "wandb_run": wandb_run,
        "log_dir": log_dir,
    }


def log_metrics(
    step: int,
    metrics: Dict[str, float],
    tb_writer=None,
    wandb_run=None,
    logger=None,
    prefix: str = "",
):
    """Log metrics to TensorBoard, WandB, and console."""
    full_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    if logger:
        logger.info(f"Step {step}: {full_metrics}")

    if tb_writer:
        for k, v in full_metrics.items():
            tb_writer.add_scalar(k, v, step)

    if wandb_run:
        wandb_run.log(full_metrics, step=step)


def close_loggers(tb_writer=None, wandb_run=None):
    """Clean up loggers."""
    if tb_writer:
        tb_writer.close()
    if wandb_run:
        wandb_run.finish()
