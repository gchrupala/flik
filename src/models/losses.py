import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple


def _gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across DDP ranks, preserving the autograd graph.

    Uses ``torch.distributed.nn.all_gather`` (autograd-aware). The local
    tensor MUST be ``.contiguous()`` — non-contiguous gradients break the
    internal ``_ReduceScatter`` backward (PyTorch #120386). Returns the
    concatenated tensor of shape ``(world_size * batch, D)``.

    When DDP is not initialized, returns the input unchanged (single-GPU
    fast path).
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    gathered = torch.distributed.nn.all_gather(tensor.contiguous())
    return torch.cat(gathered, dim=0)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE (NT‑Xent) / DCL loss for paired audio‑video embeddings.

    Under DDP, embeddings are all-gathered across ranks (autograd-preserving)
    so every rank sees the full set of negatives. The loss is computed with
    *local* semantics: each rank only computes the loss rows for its own
    ``B_local`` audio samples against the full ``N = world_size * B_local``
    video embeddings (and symmetrically for video→audio). This avoids the
    ``O(N²)`` full-logits memory and the gradient-amplification bug that
    arises when every rank computes the full global loss (OpenCLIP #1144).
    """

    def __init__(self, temperature: float = 0.07, use_dcl: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_dcl = use_dcl
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        video_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            audio_embeddings: (B_local, D) L2‑normalized, this rank's shard.
            video_embeddings: (B_local, D) L2‑normalized, this rank's shard.

        Returns:
            loss: scalar (local-loss contribution; DDP averages across ranks).
            metrics: dict with accuracy etc.
        """
        local_batch = audio_embeddings.shape[0]
        device = audio_embeddings.device

        # All-gather across DDP ranks (no-op on single GPU). Gradients flow
        # back to each rank's own embeddings only.
        all_audio = _gather_with_grad(audio_embeddings)  # (N, D)
        all_video = _gather_with_grad(video_embeddings)   # (N, D)
        global_batch = all_audio.shape[0]

        # Rank offset: this rank's samples occupy rows [rank*B_local, (rank+1)*B_local)
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        offset = rank * local_batch

        # Local logits: this rank's B_local audio vs ALL N video embeddings.
        # Shape (B_local, N). Labels are the global indices of this rank's pairs.
        logits_a = audio_embeddings @ all_video.t() / self.temperature
        logits_v = video_embeddings @ all_audio.t() / self.temperature
        labels = torch.arange(offset, offset + local_batch, device=device)

        if self.use_dcl:
            # DCL: remove the positive from the denominator.
            # The positive for row i (local index) is column labels[i] (global).
            mask = torch.zeros_like(logits_a, dtype=torch.bool)
            mask[torch.arange(local_batch, device=device), labels] = True
            neg_logits_a = logits_a.masked_fill(mask, float("-inf"))
            neg_logits_v = logits_v.masked_fill(mask, float("-inf"))
            loss_a = (-logits_a.gather(1, labels.unsqueeze(1)).squeeze(1)
                      + torch.logsumexp(neg_logits_a, dim=-1)).mean()
            loss_v = (-logits_v.gather(1, labels.unsqueeze(1)).squeeze(1)
                      + torch.logsumexp(neg_logits_v, dim=-1)).mean()
            loss = (loss_a + loss_v) / 2.0
        else:
            # Standard InfoNCE with global labels.
            loss_a = self.cross_entropy(logits_a, labels)
            loss_v = self.cross_entropy(logits_v, labels)
            loss = (loss_a + loss_v) / 2.0

        # Compute accuracy (local rows only, against global candidates)
        with torch.no_grad():
            preds_a = logits_a.argmax(dim=1)
            preds_v = logits_v.argmax(dim=1)
            acc = ((preds_a == labels).float().mean().item()
                   + (preds_v == labels).float().mean().item()) / 2.0

        metrics = {
            "contrastive_loss": loss.item(),
            "contrastive_acc": acc,
        }
        return loss, metrics


class VarianceLoss(nn.Module):
    """
    VICReg-style variance regularization.

    Penalizes embeddings whose per-dimension std (across the batch) falls below
    a target ``gamma``. This makes the collapsed solution — where every sample
    maps to the same vector (std -> 0) — a *high*-loss state instead of the
    near-zero-gradient equilibrium that pure DCL/InfoNCE settle into.

    For L2-normalized D-dim embeddings the healthy per-dim std is ~1/sqrt(D),
    so ``gamma`` defaults to that value (auto-computed from ``hidden_dim``).
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        loss = embeddings[0].new_tensor(0.0)
        for z in embeddings:
            # Gather across DDP ranks so std is measured over the global batch
            # (a per-rank std could miss cross-rank collapse). No-op on single GPU.
            z_global = _gather_with_grad(z)
            # std per dimension across the batch (biased estimator + eps)
            std = torch.sqrt(z_global.var(dim=0, unbiased=False) + self.eps)
            loss = loss + torch.mean(torch.relu(self.gamma - std))
        return loss / len(embeddings)


class MLMLoss(nn.Module):
    """
    Masked language modeling loss for wav2vec2 features.
    Predicts quantizer IDs of masked positions.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_codebook_entries: int = 320,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            logits: (batch, seq_len, num_codebook_entries).
            targets: (batch, seq_len) ground-truth target indices.
            mask: (batch, seq_len) boolean mask where True = masked positions to predict.

        Returns:
            loss: scalar.
            metrics: dict with accuracy etc.
        """
        # Only compute loss on masked positions
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device), {
                "mlm_loss": 0.0,
                "mlm_acc": 0.0,
            }

        # Gather masked predictions
        masked_logits = logits[mask]  # (num_masked, num_codebook_entries)
        masked_labels = targets[mask]  # (num_masked,)

        # Cross‑entropy loss
        loss = self.loss_fn(masked_logits, masked_labels)

        # Accuracy
        with torch.no_grad():
            preds = masked_logits.argmax(dim=-1)
            acc = (preds == masked_labels).float().mean().item()

        metrics = {
            "mlm_loss": loss.item(),
            "mlm_acc": acc,
            "mlm_masked_tokens": mask.sum().item(),
        }
        return loss, metrics


class CombinedLoss(nn.Module):
    """
    Combines contrastive and MLM losses with optional weights.
    """

    def __init__(
        self,
        contrastive_weight: float = 1.0,
        mlm_weight: float = 1.0,
        temperature: float = 0.07,
        mlm_label_smoothing: float = 0.0,
        use_dcl: bool = True,
        variance_weight: float = 0.0,
        variance_gamma: Optional[float] = None,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.mlm_weight = mlm_weight
        self.variance_weight = variance_weight
        self.contrastive = ContrastiveLoss(temperature, use_dcl=use_dcl)
        self.mlm = MLMLoss(label_smoothing=mlm_label_smoothing)
        # VICReg variance target: ~1/sqrt(D) for L2-normalized embeddings.
        if variance_gamma is None:
            variance_gamma = 1.0 / math.sqrt(hidden_dim)
        self.variance = VarianceLoss(gamma=variance_gamma)

    def forward(
        self,
        audio_embeddings: torch.Tensor,
        video_embeddings: torch.Tensor,
        mlm_logits: Optional[torch.Tensor] = None,
        mlm_targets: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns total loss and aggregated metrics.
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=audio_embeddings.device)

        # Contrastive loss
        if self.contrastive_weight > 0:
            loss_cont, metrics_cont = self.contrastive(
                audio_embeddings, video_embeddings
            )
            total_loss += self.contrastive_weight * loss_cont
            metrics.update(metrics_cont)

        # VICReg variance regularization (anti-collapse)
        if self.variance_weight > 0:
            loss_var = self.variance(audio_embeddings, video_embeddings)
            total_loss = total_loss + self.variance_weight * loss_var
            metrics["variance_loss"] = loss_var.item()

        # MLM loss
        if (
            self.mlm_weight > 0
            and mlm_logits is not None
            and mlm_targets is not None
            and mlm_mask is not None
        ):
            loss_mlm, metrics_mlm = self.mlm(mlm_logits, mlm_targets, mlm_mask)
            total_loss += self.mlm_weight * loss_mlm
            metrics.update(metrics_mlm)

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


if __name__ == "__main__":
    # Test losses with random inputs
    batch, seq_len, hidden = 4, 50, 768
    audio_emb = F.normalize(torch.randn(batch, hidden), p=2, dim=-1)
    video_emb = F.normalize(torch.randn(batch, hidden), p=2, dim=-1)

    # InfoNCE (use_dcl=False)
    cont_loss_info, metrics_info = ContrastiveLoss(use_dcl=False)(audio_emb, video_emb)
    print(
        f"InfoNCE loss: {cont_loss_info.item():.4f}, acc: {metrics_info['contrastive_acc']:.3f}"
    )

    # DCL (use_dcl=True, default)
    cont_loss_dcl, metrics_dcl = ContrastiveLoss(use_dcl=True)(audio_emb, video_emb)
    print(
        f"DCL loss:     {cont_loss_dcl.item():.4f}, acc: {metrics_dcl['contrastive_acc']:.3f}"
    )
    print(f"  (values differ: {abs(cont_loss_info.item() - cont_loss_dcl.item()) > 1e-6})")

    # MLM test
    mlm_loss = MLMLoss()
    logits = torch.randn(batch, seq_len, 320)
    targets = torch.randint(0, 320, (batch, seq_len))
    mask = torch.rand(batch, seq_len) > 0.85  # 15% masking
    loss_mlm, metrics_mlm = mlm_loss(logits, targets, mask)
    print(f"MLM loss: {loss_mlm.item():.4f}, acc: {metrics_mlm['mlm_acc']:.3f}")

    # Combined
    combined = CombinedLoss()
    total, metrics = combined(audio_emb, video_emb, logits, targets, mask)
    print(f"Combined total loss: {total.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
