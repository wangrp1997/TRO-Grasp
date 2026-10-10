#!/usr/bin/env python3
"""METHOD_V18 → Isaac 6-force eval (camera q → object frame).

Success = Isaac Gym 6-force only. Energy/contact-band are diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.hand_model import create_hand_model
from utils.v18_depth_opt import (
    flood_fill_depth,
    hand_verts_faces,
    load_frame,
    optimize_frame,
    raster_depth,
    _approach_loss,
)

YCB_RE = re.compile(r"^(\d+)(?:-[a-z])?_(.+)$")
_CODEBOOK_CACHE: dict[tuple[str, str], list[torch.Tensor]] = {}


def load_codebook(cmap: str, robot: str, max_n: int = 48) -> list[torch.Tensor]:
    key = (cmap, robot)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]
    path = os.path.join(ROOT, "data/CMapDataset_filtered/cmap_dataset.pt")
    meta = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    qs = [m[0].float() for m in meta if m[1] == cmap and m[2] == robot][:max_n]
    _CODEBOOK_CACHE[key] = qs
    return qs


def q_obj_to_cam(q_obj: torch.Tensor, T_obj_cam: np.ndarray) -> torch.Tensor:
    q = q_obj.detach().float().cpu().numpy().copy()
    Th = np.eye(4)
    Th[:3, 3] = q[:3]
    Th[:3, :3] = euler_xyz_to_R(q[3:6])
    Tc = T_obj_cam @ Th
    q[:3] = Tc[:3, 3]
    q[3:6] = R_to_euler_xyz(Tc[:3, :3])
    return torch.tensor(q, dtype=torch.float32)


def rank_codebook_image(
    hand,
    D: np.ndarray,
    K: np.ndarray,
    click: tuple[int, int],
    T_obj_cam: np.ndarray,
    q_objs: list[torch.Tensor],
    device,
    topk: int = 4,
) -> list[torch.Tensor]:
    """Re-rank object-frame codebook by image-plane overlap / approach (no mesh SDF)."""
    if not q_objs:
        return []
    M = flood_fill_depth(D, click, K)
    scale = 4
    Hs, Ws = D.shape[0] // scale, D.shape[1] // scale
    Ds = torch.as_tensor(D[::scale, ::scale], device=device)
    Mt = torch.as_tensor(M[::scale, ::scale], device=device)
    Ks = torch.as_tensor(K, device=device).clone()
    Ks[0, 0] /= scale
    Ks[1, 1] /= scale
    Ks[0, 2] /= scale
    Ks[1, 2] /= scale
    scored = []
    with torch.no_grad():
        for qo in q_objs:
            qc = q_obj_to_cam(qo, T_obj_cam).to(device)
            try:
                V, F = hand_verts_faces(hand, qc)
                dh, hm = raster_depth(V, F, Ks, Hs, Ws)
            except Exception:
                continue
            ov = float((hm & Mt).sum().item())
            if ov <= 0:
                continue
            lap = float(_approach_loss(Ds, Mt, dh, hm))
            scored.append((ov - 8.0 * lap, qo.cpu()))
    scored.sort(key=lambda x: -x[0])
    return [q for _, q in scored[:topk]]


def map_cmap(obj_name: str, cmap_set: set[str] | None = None) -> str | None:
    stem = obj_name.replace(".ply", "")
    m = YCB_RE.match(stem)
    if m:
        name = f"ycb+{m.group(2)}"
    else:
        name = f"contactdb+{stem}"
    if cmap_set is not None and name not in cmap_set:
        return None
    # require Isaac URDF
    urdf = (
        Path(ROOT)
        / "data/data_urdf/object"
        / name.split("+")[0]
        / name.split("+")[1]
        / "coacd_decomposed_object_one_link.urdf"
    )
    return name if urdf.exists() else None


def load_cmap_set(split_json: str) -> set[str]:
    data = json.load(open(split_json))
    return set(data["train"] + data["validate"])


def obj_pose_cam(extracted: Path, scene: str, fid: str, obj_index: int = 0):
    xmlp = extracted / "scenes" / scene / "realsense" / "annotations" / f"{fid}.xml"
    o = list(ET.parse(xmlp).getroot().iter("obj"))[obj_index]
    pos = np.fromstring(o.findtext("pos_in_world"), sep=" ")
    quat = np.fromstring(o.findtext("ori_in_world"), sep=" ")  # wxyz
    Rw = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rw
    T[:3, 3] = pos
    return T, o.findtext("obj_name")


def euler_xyz_to_R(rpy: np.ndarray) -> np.ndarray:
    # pytorch_kinematics floating base: roll→pitch→yaw on child frames = scipy intrinsic XYZ
    return R.from_euler("XYZ", rpy).as_matrix()


def R_to_euler_xyz(Rm: np.ndarray) -> np.ndarray:
    return R.from_matrix(Rm).as_euler("XYZ")


def q_cam_to_obj(q_cam: torch.Tensor, T_obj_cam: np.ndarray) -> torch.Tensor:
    """Floating-base q [xyz,rpy,joints] from camera frame to object frame."""
    q = q_cam.detach().float().cpu().numpy().copy()
    T_hand = np.eye(4)
    T_hand[:3, 3] = q[:3]
    T_hand[:3, :3] = euler_xyz_to_R(q[3:6])
    T_cam_obj = np.linalg.inv(T_obj_cam)
    T_ho = T_cam_obj @ T_hand
    q[:3] = T_ho[:3, 3]
    q[3:6] = R_to_euler_xyz(T_ho[:3, :3])
    return torch.tensor(q, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="/mnt/public/datasets/DexGraspNet2.0/extracted")
    ap.add_argument("--robot", default="allegro")
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--topk-isaac", type=int, default=3, help="Isaac-test top depth-opt candidates")
    ap.add_argument(
        "--codebook-topk",
        type=int,
        default=0,
        help="CMap priors count; only used with --use-codebook (privileged)",
    )
    ap.add_argument(
        "--use-codebook",
        action="store_true",
        help="PRIVILEGED ablation: category + GT T_obj CMap codebook (not for real deploy)",
    )
    ap.add_argument(
        "--no-codebook",
        action="store_true",
        help="explicitly disable codebook (default already off unless --use-codebook)",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu", type=int, default=0, help="logical GPU after CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out-dir", default="/mnt/hdd/tro_grasp/logs/v18_isaac_eval")
    ap.add_argument("--tag", default="v2", help="summary/res suffix → summary_allegro_v2.json")
    ap.add_argument(
        "--split-json",
        default=os.path.join(ROOT, "data/CMapDataset_filtered/split_train_validate_objects.json"),
    )
    ap.add_argument("--max-tries", type=int, default=200, help="max scenes to scan for mappable objs")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extracted = Path(args.extracted)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap_set = load_cmap_set(args.split_json)
    # stratified by cmap: round-robin so cracker_box does not dominate prefix
    by_cmap: dict[str, list[tuple[str, str]]] = {}
    for sc in sorted(p.name for p in (extracted / "scenes").glob("scene_*"))[: args.max_tries]:
        cam = extracted / "scenes" / sc / "realsense"
        fids = sorted(p.stem for p in (cam / "depth_gt").glob("*.png"))
        if not fids:
            continue
        fid = fids[0]
        try:
            fr0 = load_frame(extracted, sc, fid, 0)
        except Exception:
            continue
        cmap0 = map_cmap(fr0["obj_name"], cmap_set)
        if cmap0 is None:
            continue
        by_cmap.setdefault(cmap0, []).append((sc, fid))
    schedule: list[tuple[str, str, str]] = []
    keys = sorted(by_cmap.keys())
    idx = {k: 0 for k in keys}
    while len(schedule) < args.n_scenes and keys:
        progressed = False
        for k in list(keys):
            i = idx[k]
            if i >= len(by_cmap[k]):
                keys.remove(k)
                continue
            sc, fid = by_cmap[k][i]
            idx[k] = i + 1
            schedule.append((sc, fid, k))
            progressed = True
            if len(schedule) >= args.n_scenes:
                break
        if not progressed:
            break
    hand = create_hand_model(args.robot, device)

    rows = []
    n_ok = 0
    for sc, fid, _cmap_hint in schedule:
        try:
            fr = load_frame(extracted, sc, fid, 0)
        except Exception as e:
            rows.append({"scene": sc, "error": str(e)})
            continue
        cmap = map_cmap(fr["obj_name"], cmap_set)
        if cmap is None:
            continue
        T_obj, _ = obj_pose_cam(extracted, sc, fid, 0)
        out = optimize_frame(
            hand,
            fr["D"],
            fr["K"],
            fr["click"],
            steps=args.steps,
            device=device,
            n_starts=args.n_starts,
            return_all=True,
        )
        cands = out.get("candidates") or [{"q": out["q"], "score": out.get("score", 0.0)}]
        cands = cands[: max(1, args.topk_isaac)]
        q_cams = [c["q"].float() for c in cands]
        q_obj_list = [q_cam_to_obj(qc, T_obj) for qc in q_cams]
        n_opt = len(q_obj_list)
        if args.use_codebook and not args.no_codebook and args.codebook_topk > 0:
            cb = load_codebook(cmap, args.robot)
            q_cb = rank_codebook_image(
                hand,
                fr["D"],
                fr["K"],
                fr["click"],
                T_obj,
                cb,
                device,
                topk=args.codebook_topk,
            )
            q_obj_list.extend(q_cb)
        q_objs = torch.stack(q_obj_list, 0)
        pack_path = out_dir / f"{args.robot}_{args.tag}_{sc}_{fid}.pt"
        from validation.validate_utils import validate_isaac

        ok = None
        try:
            ok, _ = validate_isaac(args.robot, cmap, q_objs, gpu=args.gpu)
            if hasattr(ok, "any"):
                succ = bool(ok.any().item() if hasattr(ok.any(), "item") else ok.any())
                best_i = int(ok.float().argmax().item()) if succ else 0
            else:
                succ = bool(ok[0])
                best_i = 0
        except SystemExit as e:
            succ = False
            best_i = 0
            err = f"SystemExit:{e}"
        except Exception as e:
            succ = False
            best_i = 0
            err = str(e)
        else:
            err = None
        q_obj = q_objs[best_i]
        if best_i < n_opt:
            q_cam = q_cams[best_i]
        else:
            q_cam = q_obj_to_cam(q_obj, T_obj)
        torch.save(
            {
                "robot": args.robot,
                "scene": sc,
                "fid": fid,
                "obj_name": fr["obj_name"],
                "cmap": cmap,
                "q_cam": q_cam,
                "q_obj": q_obj,
                "q_objs": q_objs,
                "isaac_ok": ok if err is None else None,
                "hist": out["hist"],
            },
            pack_path,
        )
        n_ok += int(succ)
        row = {
            "scene": sc,
            "fid": fid,
            "obj": fr["obj_name"],
            "cmap": cmap,
            "L": out["hist"][-1]["L"],
            "L_appr": out["hist"][-1].get("L_appr"),
            "ov": out["hist"][-1].get("ov"),
            "isaac_success": succ,
            "n_cands": int(q_objs.shape[0]),
            "n_opt": n_opt,
            "best_from": "opt" if best_i < n_opt else "codebook",
            "error": err,
            "path": str(pack_path),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    summary = {
        "robot": args.robot,
        "tag": args.tag,
        "n": len(rows),
        "n_success": n_ok,
        "success_rate": (n_ok / len(rows)) if rows else 0.0,
        "steps": args.steps,
        "n_starts": args.n_starts,
        "rows": rows,
    }
    stem = f"{args.robot}_{args.tag}" if args.tag else args.robot
    man = out_dir / f"summary_{stem}.json"
    man.write_text(json.dumps(summary, indent=2))
    res = out_dir / f"res_{stem}.txt"
    with res.open("w") as f:
        f.write(f"METHOD_V18 Isaac 6-force  robot={args.robot} tag={args.tag}\n")
        f.write(f"n={len(rows)}  success={n_ok}  rate={summary['success_rate']:.4f}\n")
        for r in rows:
            if "isaac_success" in r:
                f.write(f"{r['scene']} {r.get('cmap')} {int(bool(r['isaac_success']))}\n")
    print(json.dumps({k: summary[k] for k in ("robot", "tag", "n", "n_success", "success_rate")}), flush=True)


if __name__ == "__main__":
    main()
