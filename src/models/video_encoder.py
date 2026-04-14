import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, VideoMAEConfig
from typing import Optional, Tuple, Dict


class VideoEncoder(nn.Module):
    """
    Video encoder using VideoMAE + temporal transformer.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        hidden_dim: int = 768,
        temporal_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        freeze_videomae: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim

        # Load VideoMAE model
        if pretrained:
            self.videomae = VideoMAEModel.from_pretrained(model_name)
        else:
            config = VideoMAEConfig.from_pretrained(model_name)
            config.tubelet_size = 1  # force tubelet size 1 to match input frames
            self.videomae = VideoMAEModel(config)
        config = self.videomae.config
        # print(f"VideoEncoder config hidden_size: {config.hidden_size}")
        # print(f"VideoEncoder config: {config}")
        self.patch_size = config.patch_size
        self.num_frames = config.num_frames
        self.num_patches_per_frame = (config.image_size // config.patch_size) ** 2
        self.tubelet_size = getattr(config, "tubelet_size", 2)

        # Temporal tokens expected from tubelet embedding.
        self.temporal_tokens = self.num_frames // self.tubelet_size

        # Verify hidden dimension matches
        if hidden_dim != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, hidden_dim)
        else:
            self.proj = nn.Identity()

        # Freeze VideoMAE if requested
        if freeze_videomae:
            for param in self.videomae.parameters():
                param.requires_grad = False

        # Learnable CLS token for video segment
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Positional encoding for frames (learnable)
        self.frame_pos_embed = nn.Parameter(
            torch.randn(1, self.temporal_tokens + 1, hidden_dim)
        )

        # Temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
        )

        # Layer norm
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        video: torch.Tensor,
        return_all_frames: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            video: (batch, num_frames, 3, H, W) preprocessed video tensor.
                   H = W = 224 (VideoMAE expects 224x224).
            return_all_frames: If True, return all frame embeddings, not just CLS.

        Returns:
            Dictionary with keys:
                - "cls_embedding": (batch, hidden_dim) video‑level embedding.
                - "frame_embeddings": (batch, num_frames, hidden_dim) if return_all_frames.
                - "all_embeddings": (batch, num_frames+1, hidden_dim) CLS + frames.
        """
        batch_size, num_frames = video.shape[:2]

        # VideoMAE forward (expects batch of videos, each video = stacked frames)
        # VideoMAE's input shape: (batch, num_frames, 3, H, W)
        outputs = self.videomae(video, output_hidden_states=True)
        # last_hidden_state: (batch, temporal_tokens * num_patches_per_frame, hidden_size)
        hidden = outputs.last_hidden_state  # (B, T*N, D_videomae)
        expected_tokens = self.temporal_tokens * self.num_patches_per_frame
        if hidden.shape[1] != expected_tokens:
            raise ValueError(
                "Unexpected VideoMAE token shape: "
                f"got seq_len={hidden.shape[1]}, expected {expected_tokens} "
                f"(temporal_tokens={self.temporal_tokens}, patches/frame={self.num_patches_per_frame})"
            )

        # Reshape by tubelet-level temporal tokens.
        hidden = hidden.reshape(
            batch_size,
            self.temporal_tokens,
            self.num_patches_per_frame,
            -1,
        )
        # Average over patches to get tubelet-level embeddings.
        frame_embeddings = hidden.mean(dim=2)  # (batch, temporal_tokens, D_videomae)

        # Project to hidden_dim if needed
        frame_embeddings = self.proj(frame_embeddings)  # (batch, num_frames, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, D)
        embeddings = torch.cat([cls_tokens, frame_embeddings], dim=1)  # (batch, 1+T, D)

        # Add positional embedding
        embeddings = embeddings + self.frame_pos_embed[:, : embeddings.shape[1], :]

        # Temporal transformer
        embeddings = self.ln(embeddings)
        # No padding mask (all frames are valid)
        temporal_out = self.temporal_transformer(embeddings)  # (batch, 1+T, D)

        # Split CLS and frame embeddings
        video_cls = temporal_out[:, 0, :]  # (batch, D)
        frame_embeddings_out = temporal_out[:, 1:, :]  # (batch, T, D)

        out_dict = {
            "cls_embedding": video_cls,
            "all_embeddings": temporal_out,
        }
        out_dict["frame_embeddings"] = frame_embeddings_out

        return out_dict

    def get_num_frames(self) -> int:
        """Return expected number of input frames."""
        # VideoMAE‑base expects 16 frames
        return 16


if __name__ == "__main__":
    # Test with random input
    encoder = VideoEncoder()
    batch_size, num_frames = 2, 16
    video = torch.randn(batch_size, num_frames, 3, 224, 224)

    out = encoder(video)
    cls_emb = out["cls_embedding"]
    all_emb = out["all_embeddings"]

    print(f"VideoEncoder test:")
    print(f"  Input shape: {video.shape}")
    print(f"  CLS embedding shape: {cls_emb.shape}")
    print(f"  All embeddings shape: {all_emb.shape}")
    print(f"  Hidden dim: {cls_emb.shape[-1]}")
