#!/usr/bin/env python3
"""Build clutter observations from CMap objects (not half-space crop).

Place 2-4 objects on a table, keep a top-down hidden-point view, express the
visible scene cloud in the TARGET object frame, and pair it with an existing
CMap grasp (q, object, robot). That is occlusion-by-neighbors, not solo crop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_mesh(object_name: str) -> trimesh.Trimesh:
    a, b = object_name.split("+")
    path = os.path.join(ROOT, f"data/data_urdf/object/{a}/{b}/{b}.stl")
    return trimesh.load_mesh(path)


def table_pose(rng, xy=0.08):
    T = np.eye(4)
    T[:3, :3] = R.from_euler("z", rng.uniform(0, 2 * np.pi)).as_matrix()
    T[0, 3] = rng.uniform(-xy, xy)
    T[1, 3] = rng.uniform(-xy, xy)
    T[2, 3] = 0.0
    return T


def hidden_point_keep(pts: np.ndarray, cam: np.ndarray, n_keep: int):
    """Keep points with locally maximal depth toward the camera (coarse HPR)."""
    v = pts - cam[None, :]
    dist = np.linalg.norm(v, axis=1).clip(1e-6)
    direc = v / dist[:, None]
    # azimuth/elevation bins
    az = np.arctan2(direc[:, 1], direc[:, 0])
    el = np.arcsin(np.clip(direc[:, 2], -1, 1))
    nb = 32
    az_i = np.clip(((az + np.pi) / (2 * np.pi) * nb).astype(np.int32), 0, nb - 1)
    el_i = np.clip(((el + np.pi / 2) / np.pi * nb).astype(np.int32), 0, nb - 1)
    key = az_i * nb + el_i
    best = {}
    for i, k in enumerate(key):
        d = dist[i]
        if k not in best or d < best[k][0]:
            best[k] = (d, i)
    vis = np.array([t[1] for t in best.values()], dtype=np.int64)
    if vis.size > n_keep:
        vis = vis[np.random.choice(vis.size, n_keep, replace=False)]
    elif vis.size < n_keep:
        extra = np.random.choice(pts.shape[0], n_keep - vis.size, replace=True)
        vis = np.concatenate([vis, extra])
    return vis


def invert(T):
    inv = np.eye(4)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ T[:3, 3]
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-pts", type=int, default=512)
    ap.add_argument("--out", default="/mnt/hdd/tro_grasp/data/cmap_clutter_train.pt")
    args = ap.parse_args()

    split = json.load(
        open(os.path.join(ROOT, "data/CMapDataset_filtered/split_train_validate_objects.json"))
    )
    objects = split["train"]
    cmap = torch.load(
        "/mnt/hdd/tro_grasp/data/cmap_dataset_with_boyahand.pt", map_location="cpu"
    )["metadata"]
    by_obj = {}
    for q, obj, robot in cmap:
        if obj in objects:
            by_obj.setdefault(obj, []).append((q, robot))
    usable = [o for o in objects if o in by_obj]
    cam = np.array([0.0, 0.0, 0.45], dtype=np.float64)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(args.n):
        target = usable[i % len(usable)]
        others = [o for o in usable if o != target]
        k = int(rng.integers(1, min(3, len(others)) + 1))
        clutter = list(rng.choice(others, size=k, replace=False))
        names = [target] + clutter
        world_pts = []
        poses = {}
        for name in names:
            mesh = load_mesh(name).copy()
            T = table_pose(rng)
            mesh.apply_transform(T)
            pts, _ = mesh.sample(2048, return_index=True)
            world_pts.append(np.asarray(pts))
            poses[name] = T
        scene = np.concatenate(world_pts, axis=0)
        vis_idx = hidden_point_keep(scene, cam, args.n_pts)
        vis = scene[vis_idx]
        T_tgt = poses[target]
        vis_h = np.concatenate([vis, np.ones((vis.shape[0], 1))], axis=1)
        vis_obj = (invert(T_tgt) @ vis_h.T).T[:, :3]
        q, robot = by_obj[target][int(rng.integers(0, len(by_obj[target])))]
        rows.append(
            {
                "q": q,
                "object": target,
                "robot": robot,
                "clutter": clutter,
                "scene_pc": torch.tensor(vis_obj, dtype=torch.float32),
                "view_dir": torch.tensor(
                    invert(T_tgt)[:3, :3] @ np.array([0.0, 0.0, 1.0]),
                    dtype=torch.float32,
                ),
            }
        )
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{args.n}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"samples": rows, "n": len(rows), "kind": "cmap_clutter_hpr"}, args.out)
    print("wrote", args.out, "n", len(rows), "pc", rows[0]["scene_pc"].shape)


if __name__ == "__main__":
    main()
