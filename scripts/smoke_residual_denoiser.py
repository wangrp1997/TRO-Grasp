"""CPU smoke: freeze base + residual; drop clutter => eps_res ~ 0. Do not use GPU."""
import os
import sys

import torch
from omegaconf import OmegaConf

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.tro_graph import RobotGraph


def dummy_batch(device, n_link=17, n_pts=512):
    object_pc = torch.randn(1, n_pts, 3, device=device)
    clt = object_pc + torch.tensor([0.4, 0.0, 0.0], device=device)
    scene_pc = torch.cat([object_pc[:, : n_pts // 2], clt[:, : n_pts // 2]], dim=1)
    scene_label = torch.cat(
        [
            torch.zeros(1, n_pts // 2, dtype=torch.long, device=device),
            torch.ones(1, n_pts // 2, dtype=torch.long, device=device),
        ],
        dim=1,
    )
    target_vec = [torch.randn(n_link, 6, device=device)]
    return {
        "robot_name": ["shadowhand"],
        "object_pc": object_pc,
        "scene_pc": scene_pc,
        "scene_label": scene_label,
        "target_vec": target_vec,
    }


def main():
    cfg = OmegaConf.load(os.path.join(ROOT, "config/train_residual_clutter.yaml"))
    cfg.model.N_t_training = 1
    cfg.model.embodiment = ["shadowhand"]
    cfg.model.mode = "train"
    device = torch.device("cpu")
    print("building RobotGraph on CPU...")
    model = RobotGraph(**cfg.model).to(device)
    ckpt_path = os.path.join(ROOT, "ckpt/partial.pth")
    blob = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(blob["model_state"], strict=False)
    model.freeze_eps_base()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_base = sum(p.numel() for p in model.denoiser.parameters() if p.requires_grad)
    print("trainable", n_train, "base_trainable", n_base)
    assert n_base == 0, "eps_base must be frozen"
    assert model.use_residual and model.residual_denoiser is not None

    batch = dummy_batch(device)
    model.eval()
    with torch.no_grad():
        object_pc = batch["object_pc"]
        object_nodes, _, _, _, _, scale, centroids = model._encode_object(object_pc)
        drop_on = torch.ones(1, dtype=torch.bool)
        drop_off = torch.zeros(1, dtype=torch.bool)
        V_drop, z_drop, pt_drop = model._residual_object_graph(
            batch, object_pc, object_nodes, scale, centroids, drop_on
        )
        V_full, z_full, pt_full = model._residual_object_graph(
            batch, object_pc, object_nodes, scale, centroids, drop_off
        )
        assert int((pt_drop == 1).sum()) == 0, "drop must remove clt tokens"
        assert int((pt_full == 1).sum()) > 0, "with clutter must have clt tokens"
        assert torch.allclose(z_drop, torch.zeros_like(z_drop)), "drop z_s must be 0"

        B, L = 1, model.max_link_node
        noisy_V_R = torch.randn(B, L, 6 + model.link_embed_dim)
        t = torch.zeros(B, dtype=torch.long)
        se3 = torch.eye(4).expand(B, L, 4, 4).clone()
        se3[:, :, :3, 3] = noisy_V_R[:, :, :3]
        from utils.rotation import compute_batch_relative_se3, matrix_to_vector, vector_to_matrix

        noisy_V_R_se3 = vector_to_matrix(noisy_V_R[:, :, :6])
        e_rr = matrix_to_vector(compute_batch_relative_se3(noisy_V_R_se3, noisy_V_R_se3))
        e_or_drop = model._or_edges(noisy_V_R_se3, V_drop[:, :, :3])
        e_or_full = model._or_edges(noisy_V_R_se3, V_full[:, :, :3])
        eps_res_drop = model._eps_res(V_drop, noisy_V_R, e_or_drop, e_rr, t, z_drop, pt_drop)
        eps_res_full = model._eps_res(V_full, noisy_V_R, e_or_full, e_rr, t, z_full, pt_full)
        n_drop = float(eps_res_drop.pow(2).mean())
        n_full = float(eps_res_full.pow(2).mean())
        print("eps_res drop", n_drop, "with_clt", n_full)
        assert n_drop < 1e-8, n_drop

        model.train()
        model.p_drop_clutter = 1.0
        loss = model(batch)
        print("forward loss", {k: float(v) for k, v in loss.items()})
        assert float(loss["eps_res_norm"]) < 1e-8

        ch = object_nodes[:, :, 3]
        ident_w = torch.zeros_like(ch)
        assert torch.allclose(ch, ch[:, :1], atol=1e-5)
        print("OK residual smoke; missing_keys", len(missing), "unexpected", len(unexpected))


if __name__ == "__main__":
    main()
