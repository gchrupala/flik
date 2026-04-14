import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    InfoNCE (NT‑Xent) loss for paired audio‑video embeddings.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
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

        # Labels are the diagonal (positive pairs)
        labels = torch.arange(batch_size, device=device)

        # Symmetric loss
        loss_a = self.cross_entropy(logits, labels)
        loss_v = self.cross_entropy(logits.t(), labels)
        loss = (loss_a + loss_v) / 2.0

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        metrics = {
            "contrastive_loss": loss.item(),
            "contrastive_acc": acc,
        }
        return loss, metrics


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
        self.hidden_dim = hidden_dim
        self.num_codebook_entries = num_codebook_entries
        self.classifier = nn.Linear(hidden_dim, num_codebook_entries)
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
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.mlm_weight = mlm_weight
        self.contrastive = ContrastiveLoss(temperature)
        self.mlm = MLMLoss(label_smoothing=mlm_label_smoothing)

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

    cont_loss, metrics = ContrastiveLoss()(audio_emb, video_emb)
    print(
        f"Contrastive loss: {cont_loss.item():.4f}, acc: {metrics['contrastive_acc']:.3f}"
    )

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
