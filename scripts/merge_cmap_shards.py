#!/usr/bin/env python3
"""Merge DGN2 cmap clutter shards. Dedup keys: dgn2_scene, frame, object, robot (C01)."""
from __future__ import annotations

import os
from pathlib import Path

import torch

DATA = Path("/mnt/hdd/tro_grasp/data")
OUT = DATA / "dgn2_cmap_clutter_merged.pt"


def rows_of(p: Path):
    d = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        for k in ("samples", "info"):
            if k in d and isinstance(d[k], list):
                return d[k]
    raise TypeError(f"{p}: unexpected payload {type(d)}")


def sample_key(x):
    if not isinstance(x, dict):
        return id(x)
    return (
        x.get("dgn2_scene") or x.get("scene_id") or x.get("scene"),
        x.get("frame") or x.get("frame_id") or x.get("fid"),
        x.get("dgn2_obj") or x.get("object") or x.get("object_code"),
        x.get("robot") or x.get("robot_name"),
    )


def main():
    files = []
    base = DATA / "dgn2_cmap_clutter.pt"
    if base.exists():
        files.append(base)
    files.extend(sorted(DATA.glob("dgn2_cmap_clutter_shard*.pt")))
    all_rows, seen, uniq, sources = [], set(), [], []
    for p in files:
        r = rows_of(p)
        print(p.name, "n", len(r), flush=True)
        sources.append({"path": str(p), "n": len(r)})
        all_rows.extend(r)
    for x in all_rows:
        k = sample_key(x)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    if OUT.exists():
        os.chmod(OUT, 0o644)
    torch.save(
        {
            "samples": uniq,
            "n": len(uniq),
            "n_kept": len(uniq),
            "merged_from": sources,
            "dedup_key": ["dgn2_scene", "frame", "dgn2_obj", "robot"],
            "label_source": "CMap q only; no LEAP",
        },
        OUT,
    )
    print("wrote", OUT, "n", len(uniq), "from", len(all_rows), flush=True)


if __name__ == "__main__":
    main()
