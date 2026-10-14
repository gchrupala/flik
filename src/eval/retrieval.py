import torch
from typing import Dict, Iterable, List, Union


def retrieval_metrics(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, Union[int, float, List[bool]]]:
    """Compute bidirectional retrieval metrics with per-query hit vectors.

    Assumes the i-th audio corresponds to the i-th video and that embeddings
    are L2-normalized.

    Returns a flat dict. For each requested ``k`` and each direction in
    ``("a2v", "v2a")``:

    * ``<dir>_hits@<k>``    -- int, number of queries whose match is in top-k
    * ``<dir>_rate@<k>``    -- float, hits / n_pool (recall@k)
    * ``<dir>_lift@<k>``    -- float, rate * n_pool / k (lift over chance)
    * ``<dir>_hit_vec@<k>`` -- list[bool], one entry per pool item

    ``n_pool`` is the number of paired queries/candidates. The legacy keys
    ``<dir>_r@<k_eff>`` (with ``k_eff = min(k, n_pool)``) are also emitted so
    ``retrieval_recall_at_k`` keeps its original contract.
    """
    if audio_embeddings.ndim != 2 or video_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D: (batch, dim)")
    if audio_embeddings.shape[0] != video_embeddings.shape[0]:
        raise ValueError("Audio and video batch sizes must match")
    if audio_embeddings.shape[0] == 0:
        raise ValueError("Empty embedding batch")

    n_pool = int(audio_embeddings.shape[0])
    sim = torch.mm(audio_embeddings, video_embeddings.t())
    labels = torch.arange(n_pool, device=sim.device)

    metrics: Dict[str, Union[int, float, List[bool]]] = {"n_pool": n_pool}
    for k in ks:
        k_req = int(k)
        k_eff = min(k_req, n_pool)
        topk_a2v = sim.topk(k_eff, dim=1).indices
        topk_v2a = sim.t().topk(k_eff, dim=1).indices

        # Per-query boolean hit vector: True when the paired match is in top-k.
        hit_a2v = (topk_a2v == labels.unsqueeze(1)).any(dim=1)
        hit_v2a = (topk_v2a == labels.unsqueeze(1)).any(dim=1)

        for direction, hit in (("a2v", hit_a2v), ("v2a", hit_v2a)):
            hits = int(hit.sum().item())
            # float32 mean matches the original retrieval_recall_at_k exactly.
            rate = float(hit.float().mean().item())
            metrics[f"{direction}_hits@{k_req}"] = hits
            metrics[f"{direction}_rate@{k_req}"] = rate
            metrics[f"{direction}_lift@{k_req}"] = rate * n_pool / k_req
            metrics[f"{direction}_hit_vec@{k_req}"] = hit.tolist()
            # Legacy key (original retrieval_recall_at_k naming).
            metrics[f"{direction}_r@{k_eff}"] = rate
    return metrics


def retrieval_recall_at_k(
    audio_embeddings: torch.Tensor,
    video_embeddings: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Thin wrapper over :func:`retrieval_metrics` returning only R@K rates.

    Keeps the original key convention: R@k is keyed by ``min(k, n_pool)``.
    """
    full = retrieval_metrics(audio_embeddings, video_embeddings, ks=ks)
    n_pool = int(full["n_pool"])
    return {
        f"{direction}_r@{min(int(k), n_pool)}": float(full[f"{direction}_r@{min(int(k), n_pool)}"])
        for k in ks
        for direction in ("a2v", "v2a")
    }
