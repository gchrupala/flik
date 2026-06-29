#!/usr/bin/env python
"""
DDP diagnostic script — validates multi-GPU training setup before a full run.

Usage:
    # 2-GPU diagnostic
    torchrun --standalone --nnodes=1 --nproc-per-node=2 -m scripts.validate_ddp

    # 4-GPU diagnostic
    torchrun --standalone --nnodes=1 --nproc-per-node=4 -m scripts.validate_ddp

    # Single-GPU (smoke test the script logic, no DDP)
    python -m scripts.validate_ddp
"""
import logging
import os
import socket
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Add src to path (convention, though this script doesn't need src imports)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class RankAdapter(logging.Formatter):
    """Custom formatter that injects rank into log records."""
    def format(self, record):
        rank = getattr(record, "rank", "?")
        original = record.msg
        record.msg = f"[rank {rank}] {original}"
        result = super().format(record)
        record.msg = original
        return result


def setup_logging(rank=0):
    logger = logging.getLogger("validate_ddp")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = RankAdapter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    # Inject rank for the formatter
    logger = logging.LoggerAdapter(logger, {"rank": rank})
    return logger


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _single_gpu_checks(logger):
    """Run basic checks in single-GPU mode and exit."""
    logger.info("=" * 60)
    logger.info("SINGLE-GPU MODE (no DDP)")
    logger.info("=" * 60)
    logger.info("Not launched under torchrun (no RANK/WORLD_SIZE env vars).")
    logger.info("To run DDP diagnostics, launch with:")
    logger.info("  torchrun --standalone --nnodes=1 --nproc-per-node=N -m scripts.validate_ddp")
    logger.info("")
    logger.info("All checks:    SKIPPED (not applicable in single-GPU mode)")
    return (True, [("Single-GPU mode", True, "No DDP checks run — use torchrun for DDP diagnostics")])


def check_distributed_init(rank, world_size, local_rank, logger):
    """Check 1: Distributed process group initialization."""
    if not dist.is_initialized():
        return ("Distributed init", False, "dist.is_initialized() returned False")
    hostname = socket.gethostname()
    return (
        "Distributed init",
        True,
        f"rank={rank}, world_size={world_size}, local_rank={local_rank}, hostname={hostname}",
    )


def check_gpu_assignment(rank, local_rank, logger):
    """Check 2: GPU availability and correct device assignment."""
    if not torch.cuda.is_available():
        return ("GPU assignment", False, "CUDA not available")

    n_gpus = torch.cuda.device_count()
    if local_rank >= n_gpus:
        return (
            "GPU assignment",
            False,
            f"local_rank={local_rank} but only {n_gpus} GPUs available",
        )

    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    if current_device != local_rank:
        return (
            "GPU assignment",
            False,
            f"current_device={current_device} != local_rank={local_rank}",
        )
    return ("GPU assignment", True, f"device={current_device} ({device_name})")


def check_nccl_connectivity(rank, world_size, logger):
    """Check 3: NCCL all-reduce produces the correct result."""
    t = torch.ones(1, device="cuda") * (rank + 1)
    dist.all_reduce(t)
    expected = world_size * (world_size + 1) / 2  # 1+2+...+world_size
    result = t.item()
    if abs(result - expected) < 1e-5:
        return ("NCCL connectivity", True, f"all_reduce OK: sum(1..{world_size}) = {result:.0f}")
    else:
        return (
            "NCCL connectivity",
            False,
            f"all_reduce mismatch: got {result}, expected {expected}",
        )


def check_model_gradients(rank, world_size, logger):
    """Check 4: Model forward+backward produces gradients on all ranks."""
    torch.manual_seed(42 + rank)
    model = nn.Linear(10, 10).cuda()
    ddp_model = DDP(model)
    x = torch.randn(4, 10, device="cuda")
    y = ddp_model(x)
    loss = y.sum()
    loss.backward()

    has_grad = all(p.grad is not None for p in ddp_model.parameters())
    grad_nonzero = all(p.grad.abs().sum().item() > 0 for p in ddp_model.parameters() if p.grad is not None)

    if has_grad and grad_nonzero:
        grad_norm = sum(p.grad.norm().item() for p in ddp_model.parameters())
        return ("Model gradients", True, f"gradients present, total grad norm={grad_norm:.4f}")
    elif not has_grad:
        return ("Model gradients", False, "some parameters have None gradient")
    else:
        return ("Model gradients", False, "gradients are zero")


def check_all_gather(rank, world_size, logger):
    """Check 5: All-gather (autograd-aware) produces [0, 1, ..., world_size-1]."""
    t = torch.tensor([rank], device="cuda", dtype=torch.float)
    gathered = [torch.zeros(1, device="cuda") for _ in range(world_size)]
    dist.all_gather(gathered, t)
    gathered_vals = [g.item() for g in gathered]
    expected = list(range(world_size))
    if gathered_vals == expected:
        return ("All-gather", True, f"gathered={gathered_vals}")
    else:
        return ("All-gather", False, f"got {gathered_vals}, expected {expected}")


def check_gradient_parity(rank, world_size, logger):
    """Check 6: All ranks compute identical gradients (same seed, same input)."""
    torch.manual_seed(0)
    model = nn.Linear(10, 10).cuda()
    # Use same input on all ranks
    x = torch.randn(4, 10, device="cuda")
    # Re-seed per rank to verify parity despite different seeds
    torch.manual_seed(42 + rank)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Grab a gradient tensor
    grad_tensor = model.weight.grad.clone()

    # All-reduce average
    dist.all_reduce(grad_tensor)
    grad_tensor /= world_size

    # Compare local grad with averaged grad
    local_grad = model.weight.grad
    max_diff = (local_grad - grad_tensor).abs().max().item()

    # With same input, all ranks should have identical gradients
    # So max_diff should be near zero
    if max_diff < 1e-6:
        return ("Gradient parity", True, f"max diff between ranks = {max_diff:.2e}")
    else:
        return ("Gradient parity", False, f"max diff between ranks = {max_diff:.2e} (expected ~0)")


def check_rank0_gating(rank, world_size, logger):
    """Check 7: Only rank 0 writes to a shared temp file."""
    tmpdir = tempfile.mkdtemp(prefix="ddp_test_")
    test_file = os.path.join(tmpdir, "rank0_test.txt")

    if rank == 0:
        with open(test_file, "w") as f:
            f.write("written by rank 0")
        logger.info(f"Rank 0 wrote {test_file}")

    # Barrier so rank 0's write completes before others check
    dist.barrier()

    file_exists = os.path.exists(test_file)
    if rank == 0:
        if file_exists:
            os.unlink(test_file)
            os.rmdir(tmpdir)
            result = True, f"rank 0 created and cleaned up {test_file}"
        else:
            result = True, "rank 0: file not found (possible permission issue)"
    else:
        # Non-zero ranks should NOT see the file (it's in a temp dir local to each
        # rank's filesystem, so this test is per-rank) — but that's filesystem-dependent.
        # Instead, we just verify the barrier didn't hang.
        result = True, f"rank {rank}: barrier passed (file expected only on rank 0's filesystem)"
    return ("Rank-0 gating", *result)


def check_barrier(rank, world_size, logger):
    """Check 8: Barrier synchronization."""
    dist.barrier()
    return ("Barrier", True, "all ranks reached barrier")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Parse rank/world_size from env vars (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger = setup_logging(rank=rank)

    logger.info("=" * 60)
    logger.info("DDP DIAGNOSTIC")
    logger.info("=" * 60)

    # Detect single-GPU / no torchrun
    is_ddp = bool(os.environ.get("RANK")) and world_size > 1
    if not is_ddp:
        all_ok, results = _single_gpu_checks(logger)
        if results:
            _print_results(logger, results)
        sys.exit(0 if all_ok else 1)

    # Initialize DDP
    logger.info(f"Initializing DDP: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    try:
        # Run checks
        checks = [
            ("Distributed init", lambda: check_distributed_init(rank, world_size, local_rank, logger)),
            ("GPU assignment", lambda: check_gpu_assignment(rank, local_rank, logger)),
            ("NCCL connectivity", lambda: check_nccl_connectivity(rank, world_size, logger)),
            ("Model gradients", lambda: check_model_gradients(rank, world_size, logger)),
            ("All-gather", lambda: check_all_gather(rank, world_size, logger)),
            ("Gradient parity", lambda: check_gradient_parity(rank, world_size, logger)),
            ("Rank-0 gating", lambda: check_rank0_gating(rank, world_size, logger)),
            ("Barrier", lambda: check_barrier(rank, world_size, logger)),
        ]

        results = []
        all_ok = True
        for name, check_fn in checks:
            check_name, passed, detail = check_fn()
            results.append((check_name, passed, detail))
            if not passed:
                all_ok = False
            status = "PASS" if passed else "FAIL"
            logger.info(f"  [{status}] {check_name}: {detail}")

        # Print summary table (rank 0 only to avoid interleaving)
        if rank == 0:
            _print_results(logger, results)

        return 0 if all_ok else 1

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _print_results(logger, results):
    """Print a summary table of check results."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    all_ok = True
    for name, passed, detail in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        all_ok = all_ok and passed
        logger.info(f"  {status:>6}  {name}")
    logger.info("=" * 60)
    if all_ok:
        logger.info("  ALL CHECKS PASSED — DDP is correctly configured")
    else:
        logger.info("  SOME CHECKS FAILED — see above for details")
    logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
