import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import random

from .dual_encoder import DualEncoder
from .cross_modal import CrossModalTransformer


class FlikModel(nn.Module):
    """
    Video‑grounded speech representation learning model.
    Architecture:
        1. Dual encoder (wav2vec2 + VideoMAE) for contrastive embeddings.
        2. Cross‑attention transformer (audio queries, video key‑value).
        3. MLM head on contextualized audio features.
    """

    def __init__(
        self,
        audio_model_name: str = "facebook/wav2vec2-base",
        video_model_name: str = "MCG-NJU/videomae-base",
        hidden_dim: int = 768,
        audio_feature_layer: int = 7,
        temporal_layers: int = 2,
        cross_attention_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        freeze_audio: bool = False,
        freeze_video: bool = False,
        pretrained_audio: bool = True,
        pretrained_video: bool = True,
        use_grounded_masked_prediction: bool = False,
        mlm_mask_prob: float = 0.15,
        mlm_mask_length: int = 1,
        num_codebook_entries: int = 320,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_grounded_masked_prediction = use_grounded_masked_prediction
        self.mlm_mask_prob = mlm_mask_prob
        self.mlm_mask_length = mlm_mask_length
        self.num_codebook_entries = num_codebook_entries

        # Dual encoder (contrastive embeddings)
        self.dual_encoder = DualEncoder(
            audio_model_name=audio_model_name,
            video_model_name=video_model_name,
            hidden_dim=hidden_dim,
            audio_feature_layer=audio_feature_layer,
            temporal_layers=temporal_layers,
            dropout=dropout,
            freeze_audio=freeze_audio,
            freeze_video=freeze_video,
            pretrained_audio=pretrained_audio,
            pretrained_video=pretrained_video,
        )

        # Cross‑attention transformer (audio → video)
        self.cross_modal = CrossModalTransformer(
            hidden_dim=hidden_dim,
            num_layers=cross_attention_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # Grounded masked prediction head and frozen target projection.
        self.mlm_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codebook_entries),
        )
        self.target_projection = nn.Linear(hidden_dim, num_codebook_entries)
        for param in self.target_projection.parameters():
            param.requires_grad = False

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        audio_padding_mask: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio: (batch, 1, seq_len) raw waveform.
            video: (batch, num_frames, 3, H, W) preprocessed video.
            audio_padding_mask: (batch, seq_len) boolean mask (True = padded).
            mlm_mask: (batch, seq_len_f) optional boolean mask for MLM (True = masked).
                If None, random masking is applied.
            return_features: If True, also return intermediate features.

        Returns:
            Dictionary with keys:
                - "audio_embedding": (batch, hidden_dim) for contrastive loss.
                - "video_embedding": (batch, hidden_dim) for contrastive loss.
                - "mlm_logits": (batch, seq_len_f, num_codebook_entries), optional.
                - "mlm_targets": (batch, seq_len_f), optional.
                - "mlm_mask": (batch, seq_len_f), optional boolean mask (True = masked).
                - (optional) "audio_features": raw audio features.
                - (optional) "video_frame_features": raw video frame features.
        """
        # Step 1: Extract audio features and quantizer IDs via audio encoder
        audio_out = self.dual_encoder.audio_encoder(
            audio,
            padding_mask=audio_padding_mask,
        )
        audio_features = audio_out["features"]  # (batch, seq_len_f, D)
        audio_feature_padding_mask = audio_out["padding_mask"]  # (batch, seq_len_f)

        # Step 2: Extract video frame features via video encoder
        video_out = self.dual_encoder.video_encoder(video)
        video_frame_features = video_out["frame_embeddings"]  # (batch, num_frames, D)
        video_feature_padding_mask = None  # video frames are all valid

        # Step 3: Dual encoder embeddings (pooled) for contrastive loss
        dual_out = self.dual_encoder(
            audio,
            video,
            audio_padding_mask=audio_padding_mask,
            return_features=False,
        )
        audio_embedding = dual_out["audio_embedding"]
        video_embedding = dual_out["video_embedding"]

        # Prepare output dictionary
        out = {
            "audio_embedding": audio_embedding,
            "video_embedding": video_embedding,
        }

        if self.use_grounded_masked_prediction:
            # Step 4: Mask audio features before cross-modal conditioning.
            if mlm_mask is None:
                mlm_mask = self._create_mlm_mask(
                    audio_features.shape[:2],
                    audio_feature_padding_mask,
                    self.mlm_mask_prob,
                    self.mlm_mask_length,
                    device=audio.device,
                )

            masked_audio_features = audio_features.clone()
            masked_audio_features[mlm_mask] = 0.0

            # Step 5: Cross‑attention (audio queries, video key‑value)
            cross_out = self.cross_modal(
                masked_audio_features,
                video_frame_features,
                audio_padding_mask=audio_feature_padding_mask,
                video_padding_mask=video_feature_padding_mask,
            )
            audio_contextualized = cross_out["audio_contextualized"]

            # Step 6: Predict teacher-derived target bins at masked positions.
            mlm_logits = self.compute_mlm_logits(audio_contextualized)
            with torch.no_grad():
                teacher_logits = self.target_projection(audio_features.detach())
                mlm_targets = teacher_logits.argmax(dim=-1)

            out.update(
                {
                    "mlm_logits": mlm_logits,
                    "mlm_targets": mlm_targets,
                    "mlm_mask": mlm_mask,
                }
            )

        if return_features:
            out.update(
                {
                    "audio_features": audio_features,
                    "video_frame_features": video_frame_features,
                }
            )

        return out

    def encode_audio(
        self, audio: torch.Tensor, audio_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode audio to normalized embedding (for retrieval)."""
        return self.dual_encoder.encode_audio(audio, audio_padding_mask)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video to normalized embedding (for retrieval)."""
        return self.dual_encoder.encode_video(video)

    def compute_mlm_logits(self, audio_contextualized: torch.Tensor) -> torch.Tensor:
        """Compute logits for quantizer ID prediction."""
        return self.mlm_head(
            audio_contextualized
        )  # (batch, seq_len, num_codebook_entries)

    @staticmethod
    def _create_mlm_mask(
        shape: Tuple[int, int],
        padding_mask: Optional[torch.Tensor],
        mask_prob: float,
        mask_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create boolean mask for MLM.
        Args:
            shape: (batch_size, seq_len)
            padding_mask: (batch, seq_len) where True = padded position.
            mask_prob: probability of masking a token.
            mask_length: length of each mask span (1 = single token).
            device: output device.
        Returns:
            mask: (batch, seq_len) boolean where True = masked position.
        """
        batch_size, seq_len = shape
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        if mask_prob <= 0:
            return mask

        # Determine number of tokens to mask per sequence
        if padding_mask is not None:
            # Only consider non‑padded tokens
            num_valid = (~padding_mask).sum(dim=1)  # (batch,)
            num_to_mask = (num_valid.float() * mask_prob).long()
        else:
            num_valid = seq_len
            num_to_mask = int(seq_len * mask_prob)

        # For each sequence, randomly select start positions
        for b in range(batch_size):
            if padding_mask is not None:
                valid_indices = torch.where(~padding_mask[b])[0].tolist()
            else:
                valid_indices = list(range(seq_len))

            n = (
                int(num_to_mask)
                if isinstance(num_to_mask, int)
                else int(num_to_mask[b].item())
            )
            if n <= 0 or len(valid_indices) == 0:
                continue

            # Sample start positions without replacement
            starts = random.sample(valid_indices, min(n, len(valid_indices)))
            for start in starts:
                end = min(start + mask_length, seq_len)
                mask[b, start:end] = True

        return mask


if __name__ == "__main__":
    # Test with dummy data
    model = FlikModel(use_grounded_masked_prediction=False)
    batch_size = 2
    audio = torch.randn(batch_size, 1, 16000)  # 1 second at 16kHz
    video = torch.randn(batch_size, 16, 3, 224, 224)
    audio_padding_mask = torch.zeros(batch_size, 16000, dtype=torch.bool)

    out = model(audio, video, audio_padding_mask, return_features=True)
    print(f"FlikModel test:")
    print(f"  Audio embedding shape: {out['audio_embedding'].shape}")
    print(f"  Video embedding shape: {out['video_embedding'].shape}")
    print(f"  Grounded masked prediction enabled: {'mlm_logits' in out}")

    # Test encode methods
    audio_emb = model.encode_audio(audio)
    video_emb = model.encode_video(video)
    print(f"  Encode audio shape: {audio_emb.shape}")
    print(f"  Encode video shape: {video_emb.shape}")
