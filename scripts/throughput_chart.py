#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Draws docs/images/throughput.svg (and .png) from the throughput table in docs/BENCHMARKS.md,
so the chart on the project page always shows the numbers the document states.

    python3 scripts/throughput_chart.py            # after updating the table in BENCHMARKS.md
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "BENCHMARKS.md"
OUT = ROOT / "docs" / "images" / "throughput"


def read_table():
    """Returns {(profile, backend): [gpu1, gpu2, both]}, the GPU column names, and the CPU rows."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## Throughput")
    rows = []
    for line in text[start:].splitlines():
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break  # the first table after the heading only
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    gpus = header[2:]  # e.g. RTX 3090, RTX 5060 Ti, both
    data, cpu = {}, {}
    for row in rows[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        profile = cells[0].split()[0]  # "4K" / "2K"
        if cells[1].startswith("CPU"):
            cpu[profile] = (cells[1], float(cells[-1]))
            continue
        vals = [float(v) for v in cells[2:]]
        data[(profile, cells[1])] = vals
    return data, gpus, cpu


# Keep in step with the "Machine:" line in BENCHMARKS.md.
MACHINE = "AMD Ryzen 9 5900X (12 cores / 24 threads)  ·  NVIDIA RTX 3090 24 GB + RTX 5060 Ti 16 GB  ·  driver 615.78, CUDA 13.4"


def main():
    data, gpus, cpu = read_table()
    profiles = ["4K", "2K"]
    backends = ["CUDA", "Vulkan"]
    groups = [(p, b) for p in profiles for b in backends]
    colours = {"RTX 3090": "#4c72b0", "RTX 5060 Ti": "#55a868", "both": "#dd8452"}

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    width = 0.26
    for gi, (p, b) in enumerate(groups):
        vals = data[(p, b)]
        for k, (gpu, v) in enumerate(zip(gpus, vals)):
            x = gi + (k - 1) * width
            label = gpu if gi == 0 else None
            ax.bar(x, v, width, color=colours.get(gpu, "#888"), label=label, zorder=3)
            ax.text(x, v + 4, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    for (p, (name, v)), style in zip(cpu.items(), (":", "--")):
        ax.axhline(v, color="#666", linestyle=style, linewidth=1, zorder=2,
                   label=f"CPU {p}: {v:.1f} fps ({name.split(',')[-1].strip()})")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{p} {b}" for p, b in groups])
    ax.set_ylabel("frames per second")
    fig.suptitle("GAIL throughput: DCI JPEG 2000 at 230 Mbit/s, 24 fps (gail-bench, frames preloaded)", fontsize=11,
                 y=0.98)
    ax.set_title(MACHINE, fontsize=9, color="#444", pad=6)
    ax.grid(axis="y", color="#ddd", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, max(max(v) for v in data.values()) * 1.15)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".svg"))
    fig.savefig(OUT.with_suffix(".png"))
    print(f"wrote {OUT}.svg and .png")


if __name__ == "__main__":
    sys.exit(main())
