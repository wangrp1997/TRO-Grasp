"""Sanity: planar patch ident >> noisy sphere ident."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.kanatani_reliability import patch_plane_reliability, plane_observation_ident_loss


def main():
    torch.manual_seed(0)
    plane = torch.rand(1, 512, 2)
    plane = torch.cat([plane, torch.zeros(1, 512, 1)], dim=-1)
    sphere = torch.randn(1, 512, 3)
    sphere = sphere / sphere.norm(dim=-1, keepdim=True)
    centers = torch.zeros(1, 8, 3)
    ident_p, _ = patch_plane_reliability(plane, centers)
    ident_s, n_s = patch_plane_reliability(sphere, centers)
    links = torch.linspace(-0.02, 0.02, 6).view(1, 6, 1).repeat(1, 1, 3)
    links[..., 2] = 0.0
    ident_w, ident_n = patch_plane_reliability(plane, centers)
    loss_p = plane_observation_ident_loss(links, centers, ident_w, ident_n)
    print(f"plane_ident {ident_p.mean().item():.4f}")
    print(f"sphere_ident {ident_s.mean().item():.4f}")
    print(f"plane_loss {loss_p.item():.4f}")
    assert ident_p.mean() > ident_s.mean() + 0.2, (ident_p.mean().item(), ident_s.mean().item())
    print("OK")


if __name__ == "__main__":
    main()
