#!/usr/bin/env python3
"""Isaac 6-force eval stub for METHOD_V18.

Does not run sim unless --run-isaac is set AND GPU is free.
Default: export optimized q packs for later batch eval via validate_isaac.

Success definition (contract): Isaac Gym 6-force only.
Contact-band / energy drop are diagnostics, never success.
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
from utils.v18_depth_opt import load_frame, optimize_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="/mnt/public/datasets/DexGraspNet2.0/extracted")
    ap.add_argument("--robot", default="allegro")
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="/mnt/hdd/tro_grasp/logs/v18_isaac_queue")
    ap.add_argument("--run-isaac", action="store_true", help="call validate_isaac (needs free GPU)")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extracted = Path(args.extracted)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p.name for p in (extracted / "scenes").glob("scene_*"))[: args.n_scenes]
    hand = create_hand_model(args.robot, device)
    manifest = []

    for sc in scenes:
        cam = extracted / "scenes" / sc / "realsense"
        fids = sorted(p.stem for p in (cam / "depth_gt").glob("*.png"))
        if not fids:
            continue
        fr = load_frame(extracted, sc, fids[0], 0)
        out = optimize_frame(
            hand, fr["D"], fr["K"], fr["click"], steps=args.steps, device=device
        )
        q = out["q"].float()
        pack = {
            "robot": args.robot,
            "scene": sc,
            "fid": fids[0],
            "obj_name": fr["obj_name"],
            "obj_id": fr["obj_id"],
            "q": q,
            "note": "success = Isaac 6-force only; energy is diagnostic",
        }
        path = out_dir / f"{args.robot}_{sc}_{fids[0]}.pt"
        torch.save(pack, path)
        entry = {
            "path": str(path),
            "scene": sc,
            "obj": fr["obj_name"],
            "L": out["hist"][-1]["L"],
            "isaac_success": None,
        }
        if args.run_isaac:
            from validation.validate_utils import validate_isaac

            # object_name convention in TRO: stem without .ply
            obj = fr["obj_name"].replace(".ply", "")
            ok, _ = validate_isaac(args.robot, obj, q.unsqueeze(0), gpu=args.gpu)
            entry["isaac_success"] = bool(ok[0].item() if hasattr(ok[0], "item") else ok[0])
        manifest.append(entry)
        print(json.dumps(entry), flush=True)

    man_path = out_dir / f"manifest_{args.robot}.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {man_path} n={len(manifest)} run_isaac={args.run_isaac}", flush=True)


if __name__ == "__main__":
    main()
