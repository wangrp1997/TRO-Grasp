#!/usr/bin/env python3
"""v18 single-scene smoke — thin wrapper over utils.v18_depth_opt (METHOD_V18).

CUDA_VISIBLE_DEVICES=0 recommended. No TRO training.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.hand_model import create_hand_model
from utils.v18_depth_opt import flood_fill_depth, load_frame, optimize_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="/mnt/public/datasets/DexGraspNet2.0/extracted")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--fid", default=None)
    ap.add_argument("--robot", default="allegro")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/mnt/hdd/tro_grasp/logs/v18_smoke/single.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extracted = Path(args.extracted)
    cam = extracted / "scenes" / args.scene / "realsense"
    fid = args.fid or sorted(p.stem for p in (cam / "depth_gt").glob("*.png"))[0]
    fr = load_frame(extracted, args.scene, fid, 0)
    M = flood_fill_depth(fr["D"], fr["click"], fr["K"])
    inter = int((M & fr["gt"]).sum())
    hand = create_hand_model(args.robot, device)
    out = optimize_frame(hand, fr["D"], fr["K"], fr["click"], steps=args.steps, device=device)
    last = out["hist"][-1]
    row = {
        "scene": args.scene,
        "fid": fid,
        "obj": fr["obj_name"],
        "flood_n": out["M_tgt_n"],
        "gt_n": int(fr["gt"].sum()),
        "prec": inter / max(1, int(M.sum())),
        "rec": inter / max(1, int(fr["gt"].sum())),
        "hist": out["hist"],
        "L0": out["hist"][0]["L"],
        "L": last["L"],
        "L_clt": last["L_clt"],
        "L_tgt": last["L_tgt"],
        "L_lim": last.get("L_lim"),
        "hand_pix": last["hand_pix"],
        "ok_img_drop": (last["L_clt"] + last["L_tgt"])
        < 0.5 * (out["hist"][0]["L_clt"] + out["hist"][0]["L_tgt"] + 1e-6),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
