"""Gate: object nodes must keep TRO global scale, not ident."""
import os
import sys

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.tro_graph import RobotGraph


def main():
    cfg = OmegaConf.load(os.path.join(ROOT, "config/train.yaml"))
    cfg.model.mode = "test"
    cfg.model.inference_config = {
        "inference_mode": "unconditioned",
        "collision_guidance": {"enabled": False},
    }
    model = RobotGraph(**cfg.model)
    pc = torch.randn(2, 512, 3)
    nodes, _, _, ident_w, _, scale, _ = model._encode_object(pc)
    assert nodes.shape[-1] == 68, nodes.shape
    ch = nodes[:, :, 3]
    # scale is global per batch item, constant across patches
    assert torch.allclose(ch, ch[:, :1], atol=1e-5), "scale channel must be constant over patches"
    assert not torch.allclose(ch, ident_w, atol=1e-3), "scale channel must not equal ident"
    print("OK encode scale channel", float(scale[0]), float(ch[0, 0]), "ident_mean", float(ident_w.mean()))


if __name__ == "__main__":
    main()
