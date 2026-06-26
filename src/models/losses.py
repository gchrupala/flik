import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    InfoNCE (NT‑Xent) loss for paired audio‑video embeddings.
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
            audio_embeddings: (batch, D) L2‑normalized.
            video_embeddings: (batch, D) L2‑normalized.

        Returns:
            loss: scalar.
            metrics: dict with accuracy etc.
        """
        batch_size = audio_embeddings.shape[0]
        device = audio_embeddings.device

        # Cosine similarity matrix
        logits = (
            torch.mm(audio_embeddings, video_embeddings.t()) / self.temperature
        )  # (batch, batch)

        if self.use_dcl:
            # DCL: remove positive from denominator
            # L_DCL = -s_ii/τ + logsumexp(s_{i,j≠i}/τ)
            mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            neg_logits = logits.masked_fill(mask, float("-inf"))
            loss_a = (-logits.diag() + torch.logsumexp(neg_logits, dim=-1)).mean()
            loss_v = (-logits.diag() + torch.logsumexp(neg_logits.t(), dim=-1)).mean()
            loss = (loss_a + loss_v) / 2.0
        else:
            # Standard InfoNCE
            labels = torch.arange(batch_size, device=device)
            loss_a = self.cross_entropy(logits, labels)
            loss_v = self.cross_entropy(logits.t(), labels)
            loss = (loss_a + loss_v) / 2.0

        # Compute accuracy
        with torch.no_grad():
            labels = torch.arange(batch_size, device=device)
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

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
            # std per dimension across the batch (biased estimator + eps)
            std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
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
