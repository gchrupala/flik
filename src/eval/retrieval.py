import torch
from typing import Dict, Iterable


def retrieval_recall_at_k(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute retrieval recall@k for paired embeddings.

    Assumes ith audio corresponds to ith video.
    Embeddings are expected to be L2-normalized.
    """
    if audio_embeddings.ndim != 2 or video_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D: (batch, dim)")
    if audio_embeddings.shape[0] != video_embeddings.shape[0]:
        raise ValueError("Audio and video batch sizes must match")

    batch_size = audio_embeddings.shape[0]
    sim = torch.mm(audio_embeddings, video_embeddings.t())
    labels = torch.arange(batch_size, device=sim.device)

    metrics: Dict[str, float] = {}
    for k in ks:
        k_eff = min(int(k), batch_size)
        topk_a2v = sim.topk(k_eff, dim=1).indices
        topk_v2a = sim.t().topk(k_eff, dim=1).indices

        a2v_hit = (topk_a2v == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        v2a_hit = (topk_v2a == labels.unsqueeze(1)).any(dim=1).float().mean().item()

        metrics[f"a2v_r@{k_eff}"] = a2v_hit
        metrics[f"v2a_r@{k_eff}"] = v2a_hit

    return metrics
