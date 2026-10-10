#!/usr/bin/env python3
"""DGN2 单目 GT 深度+掩膜 -> TRO 杂堆样本；标签只用 CMap q + 邻居 mesh 穿透过滤。

不用 LEAP / 夹爪。几何对不上 ycb+ 的物体直接跳过（不拿 contactdb q 套 YCB 香蕉）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.hand_model import create_hand_model

YCB_RE = re.compile(r"^(\d+)(?:-[a-z])?_(.+)$")
PENETRATE_M = 0.001
MIN_TGT = 64
MIN_CLT = 64


def invert_rt(Rcw, t):
    Rt = Rcw.T
    return Rt, -Rt @ t


def unproject(depth, K, factor):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    vs, us = np.where(depth > 0)
    z = depth[vs, us].astype(np.float64) / float(factor)
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    return np.stack([x, y, z], axis=1), vs, us


def map_cmap(obj_name, cmap_set):
    stem = obj_name.replace(".ply", "")
    m = YCB_RE.match(stem)
    if m:
        ycb = f"ycb+{m.group(2)}"
        return ycb if ycb in cmap_set else None
    contact = f"contactdb+{stem}"
    return contact if contact in cmap_set else None


def load_cmap_index(pt_path, robots):
    meta = torch.load(pt_path, map_location="cpu", weights_only=False)["metadata"]
    idx = defaultdict(list)
    for q, obj, robot in meta:
        if robot in robots:
            idx[(obj, robot)].append(q)
    return idx


def xml_T(o):
    pos = np.fromstring(o.findtext("pos_in_world"), sep=" ")
    quat = np.fromstring(o.findtext("ori_in_world"), sep=" ")
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


def neighbor_meshes(extracted, xml_objs, target_obj):
    meshes = []
    md = Path(extracted) / "meshdata"
    T_wt = xml_T(target_obj)
    T_tw = np.eye(4)
    T_tw[:3, :3], T_tw[:3, 3] = invert_rt(T_wt[:3, :3], T_wt[:3, 3])
    tid = int(target_obj.findtext("obj_id"))
    for o in xml_objs:
        oid = int(o.findtext("obj_id"))
        if oid == tid:
            continue
        ply = md / f"{oid:03d}" / "nontextured.ply"
        if not ply.exists():
            ply = md / f"{oid:03d}" / "nontextured_simplified.ply"
        if not ply.exists():
            continue
        mesh = trimesh.load_mesh(str(ply), process=False)
        mesh = mesh.copy()
        mesh.apply_transform(T_tw @ xml_T(o))
        meshes.append(mesh)
    return meshes


def hand_hits_neighbors(hand, q, neighbors):
    vis = hand.get_trimesh_q(q)["visual"]
    verts = np.asarray(vis.vertices, dtype=np.float64)
    if verts.size == 0 or not neighbors:
        return False
    verts = verts[::8]
    for mesh in neighbors:
        try:
            samples = mesh.sample(512)
        except Exception:
            samples = np.asarray(mesh.vertices)
            if len(samples) > 512:
                samples = samples[:: max(1, len(samples) // 512)]
        if samples is None or len(samples) == 0:
            continue
        d = np.min(np.linalg.norm(verts[:, None, :] - samples[None, :: max(1, len(samples)//80), :], axis=2))
        if d < 0.005:
            return True
    return False


def subsample(pts, n, rng):
    if pts.shape[0] == 0:
        return pts
    if pts.shape[0] >= n:
        return pts[rng.choice(pts.shape[0], n, replace=False)]
    extra = rng.choice(pts.shape[0], n - pts.shape[0], replace=True)
    return np.concatenate([pts, pts[extra]], axis=0)


def parse_xml(path):
    root = ET.parse(path).getroot()
    return list(root.findall("obj"))


def process_frame(scene_dir, fid, extracted, cmap_set, cmap_idx, hands, robots, rng, n_pts):
    cam = scene_dir / "realsense"
    xmlp = cam / "annotations" / f"{fid}.xml"
    labp = cam / "label_gt" / f"{fid}.png"
    depp = cam / "depth_gt" / f"{fid}.png"
    metp = cam / "meta" / f"{fid}.mat"
    if not all(p.exists() for p in (xmlp, labp, depp, metp)):
        return []
    objs = parse_xml(xmlp)
    meta = sio.loadmat(str(metp))
    K = np.array(meta["intrinsic_matrix"], dtype=np.float64)
    factor = float(np.array(meta["factor_depth"]).reshape(-1)[0])
    depth = np.array(Image.open(depp))
    label = np.array(Image.open(labp))
    xyz, vs, us = unproject(depth, K, factor)
    pix_lab = label[vs, us]
    rows = []
    for o in objs:
        oid = int(o.findtext("obj_id"))
        cmap_name = map_cmap(o.findtext("obj_name"), cmap_set)
        if cmap_name is None:
            continue
        lab_id = oid + 1
        tgt = xyz[pix_lab == lab_id]
        clt = xyz[(pix_lab > 0) & (pix_lab != lab_id)]
        if tgt.shape[0] < MIN_TGT or clt.shape[0] < MIN_CLT:
            continue
        pos = np.fromstring(o.findtext("pos_in_world"), sep=" ")
        quat = np.fromstring(o.findtext("ori_in_world"), sep=" ")
        Rw = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        Rinv, tinv = invert_rt(Rw, pos)
        tgt_o = (Rinv @ tgt.T).T + tinv
        clt_o = (Rinv @ clt.T).T + tinv
        neighbors = neighbor_meshes(extracted, objs, o)
        all_pts = np.concatenate([tgt_o, clt_o], axis=0)
        all_lab = np.concatenate(
            [np.zeros(len(tgt_o), dtype=np.int64), np.ones(len(clt_o), dtype=np.int64)]
        )
        pick = rng.choice(len(all_pts), n_pts, replace=len(all_pts) < n_pts)
        scene = all_pts[pick]
        slab = all_lab[pick]
        for robot in robots:
            qs = cmap_idx.get((cmap_name, robot), [])
            if not qs:
                continue
            hand = hands[robot]
            kept = None
            ntry = min(8, len(qs))
            picks = rng.choice(len(qs), size=ntry, replace=len(qs) < ntry)
            for i in np.atleast_1d(picks):
                q = qs[int(i)]
                if not torch.is_tensor(q):
                    q = torch.tensor(q, dtype=torch.float32)
                else:
                    q = q.float().cpu()
                if not hand_hits_neighbors(hand, q, neighbors):
                    kept = q
                    break
            if kept is None:
                continue
            rows.append(
                {
                    "q": kept if torch.is_tensor(kept) else torch.tensor(kept, dtype=torch.float32),
                    "object": cmap_name,
                    "robot": robot,
                    "scene_pc": torch.tensor(scene, dtype=torch.float32),
                    "scene_label": torch.tensor(slab, dtype=torch.long),
                    "dgn2_scene": scene_dir.name,
                    "frame": fid,
                    "dgn2_obj": o.findtext("obj_name"),
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="/mnt/public/datasets/DexGraspNet2.0/extracted")
    ap.add_argument(
        "--cmap",
        default="/mnt/hdd/tro_grasp/data/cmap_dataset_with_boyahand.pt",
    )
    ap.add_argument(
        "--out",
        default="/mnt/hdd/tro_grasp/data/dgn2_cmap_clutter.pt",
    )
    ap.add_argument("--split-json", default=os.path.join(ROOT, "data/CMapDataset_filtered/split_train_validate_objects.json"))
    ap.add_argument("--max-scenes", type=int, default=0)
    ap.add_argument("--scene-offset", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--frames-per-scene", type=int, default=1)
    ap.add_argument("--n-pts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cmap_set = set(json.load(open(args.split_json))["train"] + json.load(open(args.split_json))["validate"])
    robots = ["allegro", "barrett", "shadowhand", "boyahand"]
    cmap_idx = load_cmap_index(args.cmap, robots)
    hands = {r: create_hand_model(r, torch.device("cpu")) for r in robots}
    scenes = sorted((Path(args.extracted) / "scenes").glob("scene_*"))
    if args.scene_offset:
        scenes = scenes[args.scene_offset :]
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    rows = []
    n_before = 0
    stats = {"scenes": 0, "frames": 0, "mapped": 0, "kept": 0, "no_q": 0}
    log = Path("/mnt/hdd/tro_grasp/logs/dgn2_convert.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    for sc in scenes:
        anns = sorted((sc / "realsense" / "annotations").glob("*.xml"))
        if not anns:
            continue
        stats["scenes"] += 1
        fids = [p.stem for p in anns]
        rng.shuffle(fids)
        for fid in fids[: args.frames_per_scene]:
            stats["frames"] += 1
            try:
                got = process_frame(
                    sc, fid, args.extracted, cmap_set, cmap_idx, hands, robots, rng, args.n_pts
                )
            except Exception as e:
                log.write_text("") if not log.exists() else None
                with log.open("a") as f:
                    f.write(f"fail {sc.name} {fid} {e}\n")
                continue
            n_before += max(1, len(got))
            rows.extend(got)
            stats["kept"] = len(rows)
            if args.max_samples and len(rows) >= args.max_samples:
                rows = rows[: args.max_samples]
                break
        if args.max_samples and len(rows) >= args.max_samples:
            break
        if stats["scenes"] % 20 == 0:
            print("progress", stats, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "samples": rows,
            "n": len(rows),
            "filter_before_note": "per-target tries up to 8 CMap q; discarded if neighbor signed_distance < -1mm",
            "n_kept": len(rows),
            "stats": stats,
            "source": args.extracted,
            "label_source": "CMap q only; no LEAP",
        },
        args.out,
    )
    print("wrote", args.out, "n", len(rows), "stats", stats)


if __name__ == "__main__":
    main()
