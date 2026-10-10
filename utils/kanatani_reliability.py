"""Kanatani (ICRA 1995) plane-fitting reliability for object patches.

A local patch is identifiable as a contact plane when the smallest
eigenvalue of the point covariance is small relative to the other two.

This is observation identifiability of the contact plane, not Li-Sastry
sigma_min of the grasp wrench map (1988), and not DexGrasp-Anything
penetration/attraction guidance.
"""
import torch


def patch_plane_reliability(points, patch_xyz, knn=32, eps=1e-8):
    """
    :param points: (B, N, 3) object point cloud in the diffusion frame
    :param patch_xyz: (B, P, 3) patch centers
    :return ident: (B, P) in [0, 1], higher = more planar / identifiable
            normal: (B, P, 3) unit plane normals
    """
    bsz, n_pts, _ = points.shape
    n_patch = patch_xyz.shape[1]
    k = min(knn, n_pts)
    dist = torch.cdist(patch_xyz, points)
    idx = dist.topk(k, dim=-1, largest=False).indices
    batch_ix = torch.arange(bsz, device=points.device)[:, None, None]
    neigh = points[batch_ix, idx]
    mean = neigh.mean(dim=2, keepdim=True)
    centered = neigh - mean
    cov = torch.matmul(centered.transpose(-1, -2), centered) / max(k - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    lam = evals.clamp_min(eps)
    ident = 1.0 - lam[..., 0] / lam.sum(dim=-1)
    normal = evecs[..., 0]
    return ident, normal


def plane_observation_ident_loss(link_xyz, patch_xyz, ident, normal, contact_margin=0.008):
    """Contacts on Kanatani-identifiable planes, spread across patches.

    Debus/Dupont: contact states are useful if parameters are identifiable.
    Kanatani ident weights which patches are valid planes; we do not
    optimize Ferrari-Canny / Li-Sastry wrench span here.
    """
    offset = link_xyz.unsqueeze(2) - patch_xyz.unsqueeze(1)
    signed = (offset * normal.unsqueeze(1)).sum(-1)
    dist = signed.abs()
    margin = max(float(contact_margin), 1e-4)
    near = torch.exp(-(dist / margin).clamp_max(20.0))
    w = near * ident.unsqueeze(1)
    patch_mass = w.sum(dim=1)
    total = patch_mass.sum(dim=-1)
    p = patch_mass / total.unsqueeze(-1).clamp_min(1e-8)
    entropy = -(p * p.clamp_min(1e-8).log()).sum(-1)
    info = torch.log(total.clamp_min(1e-8)) + 0.3 * entropy
    return -info.mean()


def identifiability_collision_loss(link_xyz, patch_xyz, ident, normal, contact_margin=0.008):
    """Backward-compatible name; no penetration/attraction terms."""
    return plane_observation_ident_loss(
        link_xyz, patch_xyz, ident, normal, contact_margin=contact_margin
    )
