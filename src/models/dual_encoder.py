import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .audio_encoder import AudioEncoder
from .video_encoder import VideoEncoder


class DualEncoder(nn.Module):
    """
    Dual encoder for contrastive learning.
    Embeds audio and video into a shared space.
    """

    def __init__(
        self,
        audio_model_name: str = "facebook/wav2vec2-base",
        video_model_name: str = "MCG-NJU/videomae-base",
        hidden_dim: int = 768,
        audio_feature_layer: int = 7,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        freeze_audio: bool = False,
        freeze_video: bool = False,
        pretrained_audio: bool = True,
        pretrained_video: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Audio encoder
        self.audio_encoder = AudioEncoder(
            model_name=audio_model_name,
            feature_layer=audio_feature_layer,
            hidden_dim=hidden_dim,
            freeze=freeze_audio,
            pretrained=pretrained_audio,
        )

        # Video encoder
        self.video_encoder = VideoEncoder(
            model_name=video_model_name,
            hidden_dim=hidden_dim,
            temporal_layers=temporal_layers,
            dropout=dropout,
            freeze_videomae=freeze_video,
            pretrained=pretrained_video,
        )

        # Projection heads (FaST-VGS recipe: 768→1536→768)
        proj_hidden = hidden_dim * 2  # 1536 for hidden_dim=768
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_hidden),   # 768 → 1536
            nn.GELU(),
            nn.Linear(proj_hidden, hidden_dim),   # 1536 → 768
        )
        self.video_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, proj_hidden),   # 768 → 1536
            nn.GELU(),
            nn.Linear(proj_hidden, hidden_dim),   # 1536 → 768
        )

    @staticmethod
    def _pool_audio_features(
        audio_features: torch.Tensor,
        feature_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if feature_padding_mask is not None:
            mask = ~feature_padding_mask  # True = valid positions
            return (audio_features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        return audio_features.mean(dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_padding_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio: (batch, 1, seq_len) raw waveform.
            video: (batch, num_frames, 3, H, W) preprocessed video.
            audio_padding_mask: (batch, seq_len) boolean mask.
            return_features: If True, also return intermediate features.

        Returns:
            Dictionary with keys:
                - "audio_embedding": (batch, hidden_dim)
                - "video_embedding": (batch, hidden_dim)
                - (optional) "audio_features": (batch, seq_len_f, hidden_dim)
                - (optional) "video_frame_features": (batch, num_frames, hidden_dim)
        """
        # Audio encoding
        audio_out = self.audio_encoder(audio, audio_padding_mask)
        audio_features = audio_out["features"]  # (batch, seq_len_f, D)

        # Pool audio features (mean over valid time steps)
        feature_padding_mask = audio_out.get("padding_mask")
        audio_pooled = self._pool_audio_features(audio_features, feature_padding_mask)

        # Video encoding
        video_out = self.video_encoder(video)
        video_pooled = video_out["cls_embedding"]  # (batch, D)

        # Project embeddings
        audio_embedding = self.audio_proj(audio_pooled)
        video_embedding = self.video_proj(video_pooled)

        # L2 normalize for contrastive loss
        audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)
        video_embedding = F.normalize(video_embedding, p=2, dim=-1)

        out = {
            "audio_embedding": audio_embedding,
            "video_embedding": video_embedding,
        }
        if return_features:
            out.update(
                {
                    "audio_features": audio_features,
                    "audio_feature_padding_mask": feature_padding_mask,
                    "video_frame_features": video_out.get("frame_embeddings"),
                }
            )

        return out

    def encode_audio(
        self,
        audio: torch.Tensor,
        audio_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode audio only, returning normalized embedding."""
        audio_out = self.audio_encoder(audio, audio_padding_mask)
        audio_features = audio_out["features"]
        feature_padding_mask = audio_out.get("padding_mask")
        audio_pooled = self._pool_audio_features(audio_features, feature_padding_mask)
        audio_embedding = self.audio_proj(audio_pooled)
        return F.normalize(audio_embedding, p=2, dim=-1)

    def encode_video(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Encode video only, returning normalized embedding."""
        video_out = self.video_encoder(video)
        video_pooled = video_out["cls_embedding"]
        video_embedding = self.video_proj(video_pooled)
        return F.normalize(video_embedding, p=2, dim=-1)


if __name__ == "__main__":
    # Test with random inputs
    dual = DualEncoder()
    batch_size = 2
    audio = torch.randn(batch_size, 1, 16000)  # 1 second
    video = torch.randn(batch_size, 16, 3, 224, 224)

    out = dual(audio, video)
    audio_emb = out["audio_embedding"]
    video_emb = out["video_embedding"]

    print(f"DualEncoder test:")
    print(f"  Audio embedding shape: {audio_emb.shape}")
    print(f"  Video embedding shape: {video_emb.shape}")
    print(f"  Norm audio: {audio_emb.norm(dim=-1)}")
    print(f"  Norm video: {video_emb.norm(dim=-1)}")
    print(f"  Cosine sim matrix:")
    sim = torch.mm(audio_emb, video_emb.t())
    print(sim)
