"""Visible-hemisphere approach loss for partial-observation grasps.

Kept object points have larger projection on `view_dir`. Predicted links
should sit on that same side of the object center, instead of wrapping into
the unobserved half. Does not change TRO object-node channels.
"""
import torch


def visibility_approach_loss(link_xyz, view_dir, obj_center, margin=0.0):
    """
    :param link_xyz: (B, L, 3)
    :param view_dir: (B, 3) unit, points toward the kept (visible) half
    :param obj_center: (B, 3)
    """
    view = view_dir / view_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rel = link_xyz - obj_center.unsqueeze(1)
    vis = (rel * view.unsqueeze(1)).sum(dim=-1)
    return torch.relu(-vis + margin).mean()
