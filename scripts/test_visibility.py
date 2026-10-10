"""Visibility loss and partial dataset view_dir."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.visibility_loss import visibility_approach_loss


def test_loss_sign():
    center = torch.zeros(1, 3)
    view = torch.tensor([[0.0, 0.0, 1.0]])
    vis_links = torch.tensor([[[0.0, 0.0, 0.05], [0.01, 0.0, 0.04]]])
    hid_links = torch.tensor([[[0.0, 0.0, -0.05], [0.01, 0.0, -0.04]]])
    lv = visibility_approach_loss(vis_links, view, center).item()
    lh = visibility_approach_loss(hid_links, view, center).item()
    assert lv < 1e-6, lv
    assert lh > 0.03, lh
    print("OK vis loss", lv, lh)


def test_partial_view_dir():
    from omegaconf import OmegaConf
    from dataset.CMapDataset import create_dataloader

    cfg = OmegaConf.create(
        {
            "batch_size": 2,
            "robot_names": ["shadowhand"],
            "debug_object_names": ["contactdb+apple"],
            "object_pc_type": "partial",
            "num_workers": 0,
            "dataset_path": "/mnt/hdd/tro_grasp/data/cmap_dataset_with_boyahand.pt",
        }
    )
    loader = create_dataloader(cfg, is_train=True)
    batch = next(iter(loader))
    vd = batch["view_dir"]
    assert vd.shape == (2, 3), vd.shape
    nrm = vd.norm(dim=-1)
    assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-5), nrm
    print("OK dataset view_dir", vd[0].tolist())


if __name__ == "__main__":
    test_loss_sign()
    test_partial_view_dir()
