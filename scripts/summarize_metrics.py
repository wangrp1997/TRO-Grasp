#!/usr/bin/env python3
"""Aggregate paper_unconditioned res.txt logs into markdown/CSV tables."""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOG_ROOT = Path("/mnt/hdd/tro_grasp/logs")

HANDS = ("shadowhand", "allegro", "barrett", "boyahand")

METRIC_RES = {
    "sr": re.compile(r"Total success rate:\s*([\d.eE+-]+)"),
    "diversity": re.compile(r"Total diversity:\s*([\d.eE+-]+)"),
    "time_s": re.compile(r"Grasp generation time:\s*([\d.eE+-]+)"),
    "pd": re.compile(r"Total penetration_m:\s*([\d.eE+-]+)"),
    "coverage": re.compile(r"Total contact_coverage:\s*([\d.eE+-]+)"),
    "ik_residual": re.compile(r"Total ik_residual_m:\s*([\d.eE+-]+)"),
    "limit_viol": re.compile(r"Total limit_violation:\s*([\d.eE+-]+)"),
    "ident": re.compile(r"Total contact_identifiability:\s*([\d.eE+-]+)"),
    "plane_ident": re.compile(r"Total plane_identifiability:\s*([\d.eE+-]+)"),
}

COLUMNS = [
    "method",
    "hand",
    "SR",
    "diversity",
    "time_s",
    "PD",
    "coverage",
    "IK_residual",
    "limit_viol",
    "ident",
    "plane_ident",
    "log_dir",
]


@dataclass
class Row:
    method: str
    hand: str
    log_dir: str
    values: dict[str, Optional[float]] = field(default_factory=dict)

    def cell(self, key: str) -> str:
        v = self.values.get(key)
        if v is None:
            return ""
        if key == "sr":
            return f"{v:.4f}"
        if key in ("pd", "ik_residual", "limit_viol", "ident", "plane_ident"):
            return f"{v:.6g}"
        if key == "coverage":
            return f"{v:.4f}"
        if key == "diversity":
            return f"{v:.4f}"
        if key == "time_s":
            return f"{v:.4f}"
        return str(v)


def parse_res_txt(path: Path) -> dict[str, Optional[float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, Optional[float]] = {}
    for key, pat in METRIC_RES.items():
        m = pat.search(text)
        if not m:
            out[key] = None
            continue
        raw = m.group(1).rstrip(".")
        out[key] = float(raw)
    return out


def classify_metrics_dir(name: str) -> Optional[tuple[str, str]]:
    for hand in HANDS:
        if name == f"paper_unconditioned_{hand}_idguide_metrics":
            return "idguide", hand
        if name == f"paper_unconditioned_{hand}_plane":
            return "plane", hand
        if name == f"paper_unconditioned_{hand}_partial_ours":
            return "partial-ours", hand
        if name == f"paper_unconditioned_{hand}_partial":
            return "partial-TRO", hand
        if name == f"paper_unconditioned_{hand}_metrics":
            return "TRO", hand
        if name == f"paper_unconditioned_{hand}_idguide":
            return "idguide", hand
        if name == f"paper_unconditioned_{hand}" and hand == "boyahand":
            return "TRO", hand
    return None


def classify_orig_dir(name: str) -> Optional[tuple[str, str]]:
    for hand in HANDS:
        if name == f"paper_unconditioned_{hand}":
            return "TRO-orig", hand
    return None


def collect_rows(log_root: Path) -> list[Row]:
    rows: list[Row] = []

    metrics_dirs = sorted(
        p for p in log_root.iterdir()
        if p.is_dir()
        and p.name.startswith("paper_unconditioned_")
        and (
            p.name.endswith("_metrics")
            or p.name.endswith("_plane")
            or p.name.endswith("_partial")
            or p.name.endswith("_partial_ours")
        )
    )
    for d in metrics_dirs:
        meta = classify_metrics_dir(d.name)
        if meta is None:
            continue
        method, hand = meta
        res = d / "res.txt"
        if not res.is_file():
            continue
        rows.append(
            Row(method=method, hand=hand, log_dir=d.name, values=parse_res_txt(res))
        )

    for extra in (
        log_root / "paper_unconditioned_boyahand",
        log_root / "paper_unconditioned_boyahand_idguide",
    ):
        meta = classify_metrics_dir(extra.name)
        if meta is None or not (extra / "res.txt").is_file():
            continue
        method, hand = meta
        rows.append(
            Row(method=method, hand=hand, log_dir=extra.name, values=parse_res_txt(extra / "res.txt"))
        )

    for hand in HANDS:
        if hand == "boyahand":
            continue
        d = log_root / f"paper_unconditioned_{hand}"
        res = d / "res.txt"
        if not res.is_file():
            continue
        vals = parse_res_txt(res)
        # TRO-orig: SR-only (drop extra metrics from older full runs if any)
        for k in list(vals):
            if k != "sr":
                vals[k] = None
        rows.append(
            Row(method="TRO-orig", hand=hand, log_dir=d.name, values=vals)
        )

    method_order = {
        "TRO-orig": 0,
        "TRO": 1,
        "idguide": 2,
        "plane": 3,
        "partial-TRO": 4,
        "partial-ours": 5,
    }
    hand_order = {h: i for i, h in enumerate(HANDS)}
    rows.sort(key=lambda r: (hand_order.get(r.hand, 99), method_order.get(r.method, 99)))
    return rows


def eval_progress(log_root: Path) -> list[str]:
    """Expected paper metrics configs vs res.txt presence."""
    lines: list[str] = []
    expected = []
    for hand in HANDS:
        expected.append((f"paper_unconditioned_{hand}_plane", "plane", hand))
        if hand == "boyahand":
            expected.append((f"paper_unconditioned_{hand}", "TRO", hand))
            expected.append((f"paper_unconditioned_{hand}_idguide", "idguide", hand))
        else:
            expected.append((f"paper_unconditioned_{hand}_metrics", "TRO", hand))
            expected.append((f"paper_unconditioned_{hand}_idguide_metrics", "idguide", hand))
    for dirname, method, hand in expected:
        res = log_root / dirname / "res.txt"
        status = "DONE" if res.is_file() else "PENDING"
        lines.append(f"  [{status}] {method} / {hand} -> {dirname}/res.txt")
    return lines


def format_markdown(rows: list[Row]) -> str:
    buf = io.StringIO()
    buf.write("| method | hand | SR | diversity | time (s) | PD | coverage | IK residual | limit viol | ident | plane_ident |\n")
    buf.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for r in rows:
        buf.write(
            "| {method} | {hand} | {sr} | {div} | {time} | {pd} | {cov} | {ik} | {lim} | {id} | {pl} |\n".format(
                method=r.method,
                hand=r.hand,
                sr=r.cell("sr"),
                div=r.cell("diversity"),
                time=r.cell("time_s"),
                pd=r.cell("pd"),
                cov=r.cell("coverage"),
                ik=r.cell("ik_residual"),
                lim=r.cell("limit_viol"),
                id=r.cell("ident"),
                pl=r.cell("plane_ident"),
            )
        )
    return buf.getvalue()


def format_csv(rows: list[Row]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(COLUMNS[:-1])  # drop log_dir from csv header row per user columns
    key_map = {
        "SR": "sr",
        "diversity": "diversity",
        "time_s": "time_s",
        "PD": "pd",
        "coverage": "coverage",
        "IK_residual": "ik_residual",
        "limit_viol": "limit_viol",
        "ident": "ident",
        "plane_ident": "plane_ident",
    }
    for r in rows:
        w.writerow(
            [r.method, r.hand]
            + [r.cell(key_map[c]) for c in COLUMNS[2:-1]]
        )
    return buf.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root",
        type=Path,
        default=LOG_ROOT,
        help="Root directory containing paper_unconditioned_* log folders",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write combined report to this file (markdown + CSV + progress)",
    )
    args = parser.parse_args()
    log_root: Path = args.log_root

    rows = collect_rows(log_root)
    progress = eval_progress(log_root)
    pending = sum(1 for ln in progress if "[PENDING]" in ln)
    running_note = (
        "Eval still in progress (some metrics res.txt missing)."
        if pending
        else "All expected paper_unconditioned_*_metrics res.txt present."
    )

    report_parts = [
        "# TRO-Grasp paper unconditioned metrics\n",
        f"\n{running_note}\n",
        "\n## Eval progress (metrics runs)\n",
        "\n".join(progress),
        "\n\n## Markdown table\n\n",
        format_markdown(rows),
        "\n## CSV\n\n```csv\n",
        format_csv(rows),
        "```\n",
    ]
    report = "".join(report_parts)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
