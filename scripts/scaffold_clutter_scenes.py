#!/usr/bin/env python3
"""Scaffold: mild clutter scenes for Phase B (not full DexGraspNet 2.0).

Places 2-4 CMap objects with random SE(3) on a table plane, samples a fused
point cloud, and keeps one designated target object's grasp metadata.
Output is for later TRO conditioning; does NOT yet train a model.
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


def random_table_pose(rng: np.random.Generator, xy_span=0.12, z=0.0):
    yaw = rng.uniform(0, 2 * np.pi)
    T = np.eye(4)
    T[:3, :3] = R.from_euler("z", yaw).as_matrix()
    T[0, 3] = rng.uniform(-xy_span, xy_span)
    T[1, 3] = rng.uniform(-xy_span, xy_span)
    T[2, 3] = z
    return T


def sample_scene(object_names, target, n_pts=2048, seed=0):
    rng = np.random.default_rng(seed)
    others = [o for o in object_names if o != target]
    k = int(rng.integers(1, min(3, len(others)) + 1))
    clutter = list(rng.choice(others, size=k, replace=False))
    pcs = []
    poses = {}
    for name in [target] + clutter:
        mesh = load_mesh(name).copy()
        T = random_table_pose(rng)
        mesh.apply_transform(T)
        pts, _ = mesh.sample(n_pts // (1 + k), return_index=True)
        pcs.append(pts)
        poses[name] = T
    scene_pc = np.concatenate(pcs, axis=0)
    if scene_pc.shape[0] > n_pts:
        idx = rng.choice(scene_pc.shape[0], n_pts, replace=False)
        scene_pc = scene_pc[idx]
    return {
        "target": target,
        "clutter": clutter,
        "poses": {k: v.tolist() for k, v in poses.items()},
        "scene_pc": torch.tensor(scene_pc, dtype=torch.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=20)
    ap.add_argument(
        "--out",
        default="/mnt/hdd/tro_grasp/data/clutter_scaffold_smoke.pt",
    )
    args = ap.parse_args()
    split = json.load(
        open(os.path.join(ROOT, "data/CMapDataset_filtered/split_train_validate_objects.json"))
    )
    objects = split["train"][:20]
    scenes = []
    for i in range(args.n_scenes):
        target = objects[i % len(objects)]
        scenes.append(sample_scene(objects, target, seed=i))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"scenes": scenes, "n": len(scenes)}, args.out)
    print(f"wrote {args.out} n={len(scenes)} pts={scenes[0]['scene_pc'].shape}")


if __name__ == "__main__":
    main()
