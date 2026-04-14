import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class CrossAttentionLayer(nn.Module):
    """
    Single cross‑attention layer (audio queries, video keys/values).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_padding_mask: Optional[torch.Tensor] = None,
        video_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio: (batch, seq_len_a, D) audio features (queries).
            video: (batch, seq_len_v, D) video frame features (keys/values).
            audio_padding_mask: (batch, seq_len_a) where True = padded position.
            video_padding_mask: (batch, seq_len_v) where True = padded position.

        Returns:
            Updated audio features (batch, seq_len_a, D).
        """
        # Self‑attention residual
        attn_out, _ = self.attention(
            query=audio,
            key=video,
            value=video,
            key_padding_mask=video_padding_mask,
            need_weights=False,
        )
        audio = self.norm1(audio + self.dropout(attn_out))

        # Feed‑forward
        ff_out = self.ff(audio)
        audio = self.norm2(audio + self.dropout(ff_out))
        return audio


class CrossModalTransformer(nn.Module):
    """
    Multi‑layer cross‑attention transformer (audio→video).
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(hidden_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        audio_padding_mask: Optional[torch.Tensor] = None,
        video_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_features: (batch, seq_len_a, D)
            video_features: (batch, seq_len_v, D)
            audio_padding_mask: (batch, seq_len_a)
            video_padding_mask: (batch, seq_len_v)

        Returns:
            Dictionary with keys:
                - "audio_contextualized": (batch, seq_len_a, D)
                - "attention_weights": optional, list of attention weights.
        """
        audio = audio_features
        for layer in self.layers:
            audio = layer(audio, video_features, audio_padding_mask, video_padding_mask)

        audio = self.norm(audio)
        return {"audio_contextualized": audio}


if __name__ == "__main__":
    # Test with random tensors
    cross = CrossModalTransformer(num_layers=2)
    batch_size, seq_len_a, seq_len_v = (
        2,
        50,
        16,
    )  # audio features 50 (1s at 50Hz), video 16 frames
    hidden = 768
    audio = torch.randn(batch_size, seq_len_a, hidden)
    video = torch.randn(batch_size, seq_len_v, hidden)

    out = cross(audio, video)
    ctx = out["audio_contextualized"]
    print(f"CrossModalTransformer test:")
    print(f"  Audio input shape: {audio.shape}")
    print(f"  Video input shape: {video.shape}")
    print(f"  Contextualized audio shape: {ctx.shape}")
    print(f"  Output matches input shape: {ctx.shape == audio.shape}")
