"""Occupancy from *observed* scene points only (real-robot transferable).

No mesh interiors, no hidden-volume SDF, no CAD completion.
A depth camera + optional target mask is enough: label 0 = target visible,
1 = other visible (clutter / table). Distances are to those points.
"""
import torch
import torch.nn.functional as F


def min_dist_to_labeled(link_xyz, scene_xyz, scene_label, want, inf=10.0):
    """
    link_xyz: (B, L, 3)
    scene_xyz: (B, N, 3)
    scene_label: (B, N)  0 target, 1 clutter
    """
    B, L, _ = link_xyz.shape
    N = scene_xyz.shape[1]
    mask = scene_label == want
    diff = link_xyz.unsqueeze(2) - scene_xyz.unsqueeze(1)
    dist = diff.pow(2).sum(-1).sqrt()
    dist = dist.masked_fill(~mask.unsqueeze(1), inf)
    dmin, _ = dist.min(dim=-1)
    empty = ~mask.any(dim=-1, keepdim=True).expand(-1, L)
    return dmin.masked_fill(empty, inf)


def query_obs_occupancy(link_xyz, scene_xyz, scene_label, inf=10.0):
    """Return (B, L, 4): d_target, d_clutter, 1/(d_clutter+eps), clutter_hit."""
    d_tgt = min_dist_to_labeled(link_xyz, scene_xyz, scene_label, 0, inf=inf)
    d_clt = min_dist_to_labeled(link_xyz, scene_xyz, scene_label, 1, inf=inf)
    inv = 1.0 / (d_clt + 1e-3)
    hit = (d_clt < 0.02).float()
    return torch.stack([d_tgt, d_clt, inv, hit], dim=-1)


def clutter_penetration_loss(link_xyz, scene_xyz, scene_label, margin=0.015):
    """Penalize links that enter the *visible* clutter cloud."""
    d_clt = min_dist_to_labeled(link_xyz, scene_xyz, scene_label, 1)
    has = (scene_label == 1).any(dim=-1)
    if not has.any():
        return link_xyz.new_tensor(0.0)
    pen = F.relu(margin - d_clt)
    return pen[has].mean()


def depth_noise(pc, sigma=0.002, drop=0.1):
    """Sim-to-real: Gaussian depth jitter + random dropouts."""
    out = pc + torch.randn_like(pc) * sigma
    if drop > 0:
        keep = torch.rand(pc.shape[0], pc.shape[1], device=pc.device) > drop
        keep[:, 0] = True
        idx = keep.float().cumsum(1).long().clamp(min=1) - 1
        gathered = out.gather(1, idx.unsqueeze(-1).expand_as(out))
        fill = ~keep
        out = torch.where(fill.unsqueeze(-1), gathered, out)
    return out
