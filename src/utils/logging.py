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
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Setup logging (console + file) and optional TensorBoard/WandB.
    Returns dictionary with loggers and writers.
    """
    os.makedirs(log_dir, exist_ok=True)

    # File handler
    log_file = os.path.join(
        log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Module logger
    logger = logging.getLogger(name)

    # TensorBoard SummaryWriter
    tb_writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)
        logger.info(f"TensorBoard logging to {tb_dir}")

    # WandB run
    wandb_run = None
    if wandb:
        import wandb as wandb_lib

        wandb_run = wandb_lib.init(
            project=wandb_project or "flik-video-grounding",
            entity=wandb_entity,
            config=config,
            dir=log_dir,
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
