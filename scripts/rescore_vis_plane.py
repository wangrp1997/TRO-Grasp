"""Rescore saved vis.pt with current extra_metrics (plane_ident)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.extra_metrics import compute_grasp_metrics
from utils.hand_model import create_hand_model

LOG = "/mnt/hdd/tro_grasp/logs"
RUNS = [
    ("TRO", "shadowhand", f"{LOG}/paper_unconditioned_shadowhand_metrics/vis.pt"),
    ("idguide", "shadowhand", f"{LOG}/paper_unconditioned_shadowhand_idguide_metrics/vis.pt"),
    ("TRO", "allegro", f"{LOG}/paper_unconditioned_allegro_metrics/vis.pt"),
    ("idguide", "allegro", f"{LOG}/paper_unconditioned_allegro_idguide_metrics/vis.pt"),
    ("TRO", "barrett", f"{LOG}/paper_unconditioned_barrett_metrics/vis.pt"),
    ("idguide", "barrett", f"{LOG}/paper_unconditioned_barrett_idguide_metrics/vis.pt"),
]


def main():
    hands = {}
    lines = [
        "method,hand,SR,PD,coverage,IK,limit,wrench,plane_ident",
    ]
    for method, hand_name, path in RUNS:
        if not os.path.isfile(path):
            print("missing", path)
            continue
        if hand_name not in hands:
            hands[hand_name] = create_hand_model(hand_name, device="cpu")
        hand = hands[hand_name]
        vis = torch.load(path, map_location="cpu")
        acc = {k: [] for k in (
            "penetration_m",
            "contact_coverage",
            "ik_residual_m",
            "limit_violation",
            "contact_identifiability",
            "plane_identifiability",
        )}
        succ = []
        for item in vis:
            q = item["predict_q"]
            suc = torch.as_tensor(item["success"]).float()
            succ.append(suc.mean().item())
            m = compute_grasp_metrics(
                hand, item["object_name"], q, None, None
            )
            for k in acc:
                acc[k].append(m[k])
        n = max(len(succ), 1)

        def mean(xs):
            xs = [x for x in xs if x == x]
            return sum(xs) / max(len(xs), 1)

        row = (
            f"{method},{hand_name},{mean(succ):.4f},{mean(acc['penetration_m']):.6g},"
            f"{mean(acc['contact_coverage']):.4f},nan,{mean(acc['limit_violation']):.6g},"
            f"{mean(acc['contact_identifiability']):.6g},{mean(acc['plane_identifiability']):.6g}"
        )
        lines.append(row)
        print(row, flush=True)
    out = os.path.join(LOG, "plane_ident_baseline.csv")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
