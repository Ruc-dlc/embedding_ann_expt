#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot sensitivity analysis of distance weight w for DACL-DR.

Inputs:
    - bert-base-uncase.json
    - bge-small-en-v1.5.json

Outputs:
    - w_sensitivity_analysis.pdf
    - w_sensitivity_analysis.png

Figure design:
    - 1x2 subplots
    - x-axis: distance weight w
    - left y-axis: Top-100 hit rate (%)
    - right y-axis: Number of Distance Computations (NDC)
    - vertical dashed line at w=0.4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_results(json_path: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Load plotting data from a JSON file.

    Expected JSON format:
    {
        "w0.0": {
            "top-100": 85.6,
            "number_of_distance_computations": 15215.64,
            ...
        },
        ...
    }

    Returns:
        ws: sorted list of w values
        top100: corresponding Top-100 values
        ndc: corresponding NDC values
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, float]] = json.load(f)

    parsed = []
    for key, metrics in data.items():
        if not key.startswith("w"):
            raise ValueError(f"Unexpected key format: {key}")
        w = float(key[1:])
        if "top-100" not in metrics:
            raise KeyError(f"'top-100' not found in {key} of {json_path}")
        if "number_of_distance_computations" not in metrics:
            raise KeyError(
                f"'number_of_distance_computations' not found in {key} of {json_path}"
            )
        parsed.append(
            (
                w,
                float(metrics["top-100"]),
                float(metrics["number_of_distance_computations"]),
            )
        )

    parsed.sort(key=lambda x: x[0])

    ws = [x[0] for x in parsed]
    top100 = [x[1] for x in parsed]
    ndc = [x[2] for x in parsed]
    return ws, top100, ndc


def format_thousands(values: List[float]) -> List[str]:
    """Format float values with thousands separators and no decimals."""
    return [f"{int(round(v)):,}" for v in values]


def plot_single_panel(
    ax: plt.Axes,
    ws: List[float],
    top100: List[float],
    ndc: List[float],
    title: str,
    selected_w: float = 0.4,
) -> None:
    """
    Plot one panel with dual y-axes:
    - left axis: Top-100
    - right axis: NDC
    """
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D
    
    ax2 = ax.twinx()

    # Left axis: Top-100 hit rate
    line1 = ax.plot(
        ws,
        top100,
        marker="o",
        linewidth=1.8,
        markersize=5.5,
        color='C0',  # 明确指定颜色
        label="Top-100",
    )

    # Right axis: NDC
    line2 = ax2.plot(
        ws,
        ndc,
        marker="s",
        linewidth=1.8,
        markersize=5.0,
        linestyle="--",
        color='C1',  # 明确指定颜色，与左轴区分
        label="NDC",
    )

    # Selected w marker
    ax.axvline(
        x=selected_w,
        linestyle=":",
        linewidth=1.2,
        color="gray",
    )

    # Axis labels
    ax.set_xlabel(r"Distance weight $w$")
    ax.set_ylabel("Top-100 hit rate (%)")
    ax2.set_ylabel("NDC")

    # Title
    ax.set_title(title, pad=10)

    # Grid only on primary axis
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # X ticks
    ax.set_xticks(ws)
    ax.set_xticklabels([f"{w:.1f}" for w in ws])

    # Reasonable axis ranges with small margins
    top_margin = max(0.2, (max(top100) - min(top100)) * 0.25)
    ndc_margin = max(200.0, (max(ndc) - min(ndc)) * 0.10)

    ax.set_ylim(min(top100) - top_margin, max(top100) + top_margin)
    ax2.set_ylim(min(ndc) - ndc_margin, max(ndc) + ndc_margin)

    # Format right y-axis with thousand separators
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}" if x >= 0 else ""))

    # Annotate selected point
    if selected_w in ws:
        idx = ws.index(selected_w)
        ax.annotate(
            r"Selected $w=0.4$",
            xy=(ws[idx], top100[idx]),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
        )

    # Combined legend using custom handles to ensure dashed line appears correctly
    legend_handles = [
        Line2D([0], [0], marker='o', color='C0', linestyle='-', 
               linewidth=1.8, markersize=5.5, label='Top-100'),
        Line2D([0], [0], marker='s', color='C1', linestyle='--', 
               linewidth=1.8, markersize=5.0, label='NDC'),
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True)

def main() -> None:
    # ---------- File paths ----------
    bert_json = "bert-base-uncase.json"
    bge_json = "bge-small-en-v1.5.json"

    # ---------- Load data ----------
    bert_ws, bert_top100, bert_ndc = load_results(bert_json)
    bge_ws, bge_top100, bge_ndc = load_results(bge_json)

    # ---------- Figure style ----------
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))

    plot_single_panel(
        axes[0],
        bert_ws,
        bert_top100,
        bert_ndc,
        title="(a) bert-base-uncased",
        selected_w=0.4,
    )

    plot_single_panel(
        axes[1],
        bge_ws,
        bge_top100,
        bge_ndc,
        title="(b) bge-small-en-v1.5",
        selected_w=0.4,
    )

    fig.suptitle(
        r"Effect of distance weight $w$ on retrieval effectiveness and HNSW search cost",
        y=1.02,
        fontsize=13,
    )

    fig.tight_layout()

    # ---------- Save ----------
    out_pdf = "w_sensitivity_analysis.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")

    print(f"Saved figure to: {out_pdf}")


if __name__ == "__main__":
    main()
