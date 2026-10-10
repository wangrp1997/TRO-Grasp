#!/usr/bin/env python3
"""Retarget shadowhand training grasps to boyahand via batched Pyroki IK.

Use JAX on CPU if GPU JAX fails (CuDNN mismatch): JAX_PLATFORMS=cpu
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch
import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model
from utils.optimization import process_transform
from utils.pyroki_ik import PyrokiRetarget

DEFAULT_SRC = os.path.join(ROOT_DIR, "data/CMapDataset_filtered/cmap_dataset.pt")
DEFAULT_OUT = "/mnt/hdd/tro_grasp/data/cmap_dataset_with_boyahand.pt"
RESIDUAL_THRESH_M = 0.02


def _resolve_path(path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def shadow_link_targets(shadow_hand, q_shadow: torch.Tensor, target_links: list[str]) -> torch.Tensor:
    """Link-frame translations (L, 3) in shadow FK."""
    shadow_hand.update_status(q_shadow)
    transform = {
        name: shadow_hand.frame_status[name].get_matrix()
        for name in target_links
    }
    optim = process_transform(shadow_hand.pk_chain, transform)
    return torch.stack([optim[name][0] for name in target_links], dim=0)


def boyahand_init_q(boyahand, q_shadow: torch.Tensor) -> torch.Tensor:
    q0 = boyahand.get_canonical_q().clone()
    q0[:6] = q_shadow[:6]
    return q0


def mean_ik_residual(boyahand, q_pred: torch.Tensor, target_pos: torch.Tensor) -> float:
    _, se3 = boyahand.get_transformed_links_pc(q_pred)
    fk = se3[:, :3, 3]
    n = min(fk.shape[0], target_pos.shape[0])
    return (fk[:n] - target_pos[:n]).norm(dim=-1).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=DEFAULT_SRC)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_grasps", type=int, default=0, help="0 = all train shadow grasps")
    parser.add_argument("--residual_thresh", type=float, default=RESIDUAL_THRESH_M)
    args = parser.parse_args()

    split_path = os.path.join(ROOT_DIR, "data/CMapDataset_filtered/split_train_validate_objects.json")
    train_objects = set(json.load(open(split_path))["train"])

    src_path = _resolve_path(args.src)
    data = torch.load(src_path, map_location="cpu")
    original_metadata = list(data["metadata"])
    shadow_grasps = [
        (q, obj, robot)
        for q, obj, robot in original_metadata
        if robot == "shadowhand" and obj in train_objects
    ]
    if args.max_grasps > 0:
        shadow_grasps = shadow_grasps[: args.max_grasps]

    shadow_hand = create_hand_model("shadowhand", torch.device("cpu"))
    boyahand = create_hand_model("boyahand", torch.device("cpu"))
    target_links = list(boyahand.links_pc.keys())
    for name in target_links:
        if name not in shadow_hand.links_pc:
            raise RuntimeError(f"Missing shadow link {name} for retargeting")

    meta_json = json.load(open(os.path.join(ROOT_DIR, "data/data_urdf/robot/urdf_assets_meta.json")))
    urdf_path = os.path.join(ROOT_DIR, meta_json["urdf_path"]["boyahand"])

    ik_solver = PyrokiRetarget(urdf_path, target_links)
    batch_retarget = jax.jit(ik_solver.solve_retarget)

    boyahand_meta = []
    n_fail = 0
    n_total = len(shadow_grasps)

    for start in tqdm.tqdm(range(0, n_total, args.batch_size), desc="retarget"):
        chunk = shadow_grasps[start : start + args.batch_size]
        target_batch = []
        init_batch = []
        for q_shadow, _obj, _ in chunk:
            q_shadow = q_shadow.float()
            target_batch.append(shadow_link_targets(shadow_hand, q_shadow, target_links))
            init_batch.append(boyahand_init_q(boyahand, q_shadow))

        target_pos = torch.stack(target_batch, dim=0)
        init_q = torch.stack(init_batch, dim=0)

        target_jnp = jnp.array(target_pos.numpy())
        init_jnp = jnp.array(init_q.numpy())
        pred_jnp = batch_retarget(initial_q=init_jnp, target_pos=target_jnp)
        jax.block_until_ready(pred_jnp)
        pred_q = torch.from_numpy(np.array(pred_jnp)).float()

        for i, (_q_sh, obj, _r) in enumerate(chunk):
            resid = mean_ik_residual(boyahand, pred_q[i], target_pos[i])
            if resid > args.residual_thresh:
                n_fail += 1
                continue
            boyahand_meta.append((pred_q[i].cpu(), obj, "boyahand"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_metadata = original_metadata + boyahand_meta
    torch.save({"metadata": out_metadata}, args.out)

    n_ok = len(boyahand_meta)
    fail_rate = n_fail / n_total if n_total else 0.0
    print(f"shadow candidates: {n_total}")
    print(f"boyahand kept: {n_ok}")
    print(f"IK fail rate (residual > {args.residual_thresh}m): {fail_rate:.4f} ({n_fail}/{n_total})")
    print(f"saved: {args.out} (total metadata {len(out_metadata)})")


if __name__ == "__main__":
    main()
