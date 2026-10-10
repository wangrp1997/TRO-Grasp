"""Visible-scene occupancy of each robot link (tgt / clutter / free)."""
from __future__ import annotations

import torch


def link_occupancy(link_xyz: torch.Tensor, scene_pc: torch.Tensor, scene_label: torch.Tensor, radius: float = 0.015):
    """link_xyz [B,L,3], scene_pc [B,N,3], scene_label [B,N] 0=tgt 1=clt.

    Returns one-hot [B,L,3] = free, tgt, clt.
    """
    B, L, _ = link_xyz.shape
    d = torch.cdist(link_xyz, scene_pc)
    near = d <= radius
    lab = scene_label[:, None, :].expand(-1, L, -1)
    hit_tgt = (near & (lab == 0)).any(dim=-1)
    hit_clt = (near & (lab == 1)).any(dim=-1)
    occ = link_xyz.new_zeros(B, L, 3)
    occ[..., 2] = hit_clt.float()
    occ[..., 1] = (hit_tgt & ~hit_clt).float()
    occ[..., 0] = (~hit_tgt & ~hit_clt).float()
    return occ
