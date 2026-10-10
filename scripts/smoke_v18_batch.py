#!/usr/bin/env python3
"""Multi-scene v18 smoke. Writes JSON under /mnt/hdd/tro_grasp/logs/v18_smoke/."""
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
    ap.add_argument("--robot", default="allegro")
    ap.add_argument("--n-scenes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default="/mnt/hdd/tro_grasp/logs/v18_smoke/batch.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extracted = Path(args.extracted)
    scenes = sorted(p.name for p in (extracted / "scenes").glob("scene_*"))[: args.n_scenes]
    hand = create_hand_model(args.robot, device)
    rows = []
    for sc in scenes:
        cam = extracted / "scenes" / sc / "realsense"
        fids = sorted(p.stem for p in (cam / "depth_gt").glob("*.png"))
        if not fids:
            continue
        fid = fids[0]
        try:
            fr = load_frame(extracted, sc, fid, 0)
        except Exception as e:
            rows.append({"scene": sc, "fid": fid, "error": str(e)})
            print("fail", sc, e, flush=True)
            continue
        M = flood_fill_depth(fr["D"], fr["click"], fr["K"])
        inter = int((M & fr["gt"]).sum())
        prec = inter / max(1, int(M.sum()))
        rec = inter / max(1, int(fr["gt"].sum()))
        out = optimize_frame(
            hand, fr["D"], fr["K"], fr["click"], steps=args.steps, device=device
        )
        h0, last = out["hist"][0], out["hist"][-1]
        img0 = h0["L_clt"] + h0["L_tgt"]
        img1 = last["L_clt"] + last["L_tgt"]
        row = {
            "scene": sc,
            "fid": fid,
            "obj": fr["obj_name"],
            "flood_n": out["M_tgt_n"],
            "gt_n": int(fr["gt"].sum()),
            "prec": prec,
            "rec": rec,
            "L0": h0["L"],
            "L": last["L"],
            "L_clt": last["L_clt"],
            "L_tgt": last["L_tgt"],
            "L_lim": last.get("L_lim"),
            "L_self": last.get("L_self"),
            "hand_pix": last["hand_pix"],
            # require real image loss drop AND hand still on screen; L0≈0 or hand_pix=0 → fail
            "ok_drop": (
                img0 > 1e-3
                and img1 < img0 * 0.5
                and last["hand_pix"] > 0
            ),
            "ok_img_drop": (
                img0 > 1e-3
                and img1 < img0 * 0.5
                and last["hand_pix"] > 0
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"robot": args.robot, "rows": rows}, indent=2))
    n_ok = sum(1 for r in rows if r.get("ok_drop"))
    print(f"wrote {args.out} n={len(rows)} ok_drop={n_ok}", flush=True)


if __name__ == "__main__":
    main()
