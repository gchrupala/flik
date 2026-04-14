from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    Wraps a pretrained Wav2Vec2 model and extracts features from a specific layer.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        feature_layer: int = 7,  # 0‑based; layer 7 = 8th transformer layer
        hidden_dim: int = 768,
        freeze: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_layer = feature_layer
        self.hidden_dim = hidden_dim

        # Load pretrained model or create from config
        if pretrained:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        else:
            config = Wav2Vec2Config.from_pretrained(model_name)
            self.wav2vec2 = Wav2Vec2Model(config)
        config = self.wav2vec2.config
        self.feature_extractor = self.wav2vec2.feature_extractor
        self.feature_projection = self.wav2vec2.feature_projection
        self.encoder = self.wav2vec2.encoder

        # Verify layer index
        num_layers = config.num_hidden_layers
        if feature_layer >= num_layers:
            raise ValueError(
                f"feature_layer {feature_layer} >= num_layers {num_layers}"
            )

        # Projection to hidden_dim (if different)
        if hidden_dim != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, hidden_dim)
        else:
            self.proj = nn.Identity()

        # Freeze whole model if requested
        if freeze:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        # Compute total stride of convolutional feature extractor
        conv_stride = (
            config.conv_stride
            if hasattr(config, "conv_stride")
            else [5, 2, 2, 2, 2, 2, 2, 2]
        )
        self.stride_product = 1
        for s in conv_stride:
            self.stride_product *= s

    def forward(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        layer: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            audio: (batch, 1, seq_len) raw waveform (already normalized).
            padding_mask: (batch, seq_len) boolean mask where True = padded position.
            layer: Override self.feature_layer.

        Returns:
            Dictionary with keys:
                - "features": (batch, seq_len, hidden_dim) extracted features.
                - "padding_mask": (batch, seq_len) padding mask for feature sequence.
        """
        layer_idx = layer if layer is not None else self.feature_layer
        batch_size = audio.size(0)

        # Wav2Vec2 expects (batch, seq_len) not (batch, 1, seq_len)
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        # Forward through wav2vec2
        attention_mask = None
        if padding_mask is not None:
            # HF Wav2Vec2 uses attention mask where 1 indicates valid samples.
            attention_mask = (~padding_mask).long()

        outputs = self.wav2vec2(
            audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract features from the specified layer
        # hidden_states: tuple of (num_layers + 1) tensors, each (batch, seq_len, hidden_size)
        # Index 0 is after feature projection, 1..num_layers are encoder layers
        # We want encoder layer at index `layer_idx + 1`
        hidden_states = outputs.hidden_states
        features = hidden_states[layer_idx + 1]  # (batch, seq_len, config.hidden_size)

        # Project to hidden_dim if needed
        features = self.proj(features)

        # Compute padding mask for feature sequence (length after convolutional layers)
        if padding_mask is not None:
            # Compute valid lengths (non‑padded positions) in original audio
            valid_lengths = (~padding_mask).sum(dim=1)  # (batch,)
            # Compute corresponding feature sequence lengths after conv stack
            feat_lengths = (valid_lengths - 1) // self.stride_product + 1
            # Create feature padding mask where first `feat_lengths` positions are valid
            feature_padding_mask = torch.zeros(
                batch_size,
                features.size(1),
                device=features.device,
                dtype=torch.bool,
            )
            for i, L in enumerate(feat_lengths):
                if L < features.size(1):
                    feature_padding_mask[i, L:] = True
        else:
            feature_padding_mask = None

        return {
            "features": features,
            "padding_mask": feature_padding_mask,
        }

    def extract_features(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convenience method: returns (features, padding_mask)."""
        out = self.forward(audio, padding_mask)
        return out["features"], out["padding_mask"]

    def get_feature_rate(self) -> float:
        """Return feature frames per second from config."""
        audio_sample_rate = self.wav2vec2.config.sample_rate
        return audio_sample_rate / self.stride_product
        # For 16kHz input and stride_product=320, feature rate is 50 Hz


if __name__ == "__main__":
    # Test with random input
    encoder = AudioEncoder()
    batch_size, seq_len = 2, 16000  # 1 second at 16kHz
    audio = torch.randn(batch_size, 1, seq_len)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    out = encoder(audio, padding_mask)
    features = out["features"]
    print("AudioEncoder test:")
    print(f"  Input shape: {audio.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Hidden dim: {features.shape[-1]}")
    print(f"  Feature rate: {encoder.get_feature_rate()} Hz")
