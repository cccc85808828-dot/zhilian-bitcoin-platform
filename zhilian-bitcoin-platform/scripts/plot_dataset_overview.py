from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tthgnn_erl.ellipticpp import EllipticPPSnapshotDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-time-step class counts and illicit ratios."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/ellipticpp"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/paper_figures/图2_Elliptic++时间步类别分布及非法交易占比.png"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
        mpl.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()
    mpl.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "hatch.linewidth": 0.45,
        }
    )


def load_time_statistics(data_root: Path) -> Dict[str, np.ndarray]:
    dataset = EllipticPPSnapshotDataset(data_root, validate=True)
    time_ids: List[int] = []
    licit: List[int] = []
    illicit: List[int] = []
    unknown: List[int] = []
    for snapshot in dataset:
        labels = snapshot.labels.cpu()
        time_ids.append(int(snapshot.time_id))
        licit.append(int((labels == 0).sum()))
        illicit.append(int((labels == 1).sum()))
        unknown.append(int((labels < 0).sum()))

    values = {
        "time": np.asarray(time_ids, dtype=np.int64),
        "licit": np.asarray(licit, dtype=np.int64),
        "illicit": np.asarray(illicit, dtype=np.int64),
        "unknown": np.asarray(unknown, dtype=np.int64),
    }
    values["ratio"] = (
        100.0
        * values["illicit"]
        / np.maximum(values["licit"] + values["illicit"], 1)
    )
    if values["time"].tolist() != list(range(1, 50)):
        raise ValueError("Expected 49 consecutive Elliptic++ time steps.")
    return values


def add_split_lines(ax: plt.Axes) -> None:
    for boundary in (30.5, 35.5):
        ax.axvline(
            boundary,
            ymin=0.0,
            ymax=1.0,
            color="black",
            linestyle=(0, (4, 3)),
            linewidth=0.9,
            dash_capstyle="butt",
            clip_on=True,
            zorder=5,
        )


def format_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.25, 49.75)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49])
    ax.set_xlabel("时间步")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, linestyle="-")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.7)


def draw_count_panel(ax: plt.Axes, values: Dict[str, np.ndarray]) -> None:
    styles = (
        ("illicit", "非法交易", "#E58B8B", "o"),
        ("licit", "合法交易", "#8FC7A1", "s"),
        ("unknown", "未知交易", "#E8CE79", "^"),
    )
    for key, label, color, marker in styles:
        ax.plot(
            values["time"],
            values[key],
            color=color,
            linewidth=1.65,
            marker=marker,
            markersize=2.7,
            markerfacecolor="white",
            markeredgewidth=0.75,
            label=label,
            zorder=3,
        )
    add_split_lines(ax)
    format_axis(ax)
    ax.set_ylabel("交易数量")
    ax.set_title("（a）各时间步交易类别数量", loc="left", pad=8, fontweight="bold")
    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#C8C8C8",
        fontsize=7.4,
        borderpad=0.45,
        handlelength=2.2,
    )


def split_ratio(
    values: Dict[str, np.ndarray], first: int, last: int
) -> float:
    mask = (values["time"] >= first) & (values["time"] <= last)
    illicit = int(values["illicit"][mask].sum())
    licit = int(values["licit"][mask].sum())
    return 100.0 * illicit / (illicit + licit)


def draw_ratio_panel(ax: plt.Axes, values: Dict[str, np.ndarray]) -> None:
    ax.bar(
        values["time"],
        values["ratio"],
        width=0.76,
        facecolor="white",
        edgecolor="black",
        linewidth=0.65,
        hatch="//////",
        zorder=3,
    )
    add_split_lines(ax)
    format_axis(ax)
    ax.set_ylabel("非法交易占比（%）")
    ax.set_title("（b）各时间步非法交易占比", loc="left", pad=8, fontweight="bold")

    top = max(44.0, float(values["ratio"].max()) + 8.0)
    ax.set_ylim(0.0, top)
    stage_specs = (
        (15.5, "训练集", split_ratio(values, 1, 30)),
        (33.0, "验证集", split_ratio(values, 31, 35)),
        (42.5, "测试集", split_ratio(values, 36, 49)),
    )
    for center, name, ratio in stage_specs:
        ax.text(
            center,
            top - 0.9,
            f"{name}\n{ratio:.2f}%",
            ha="center",
            va="top",
            fontsize=7.3,
            fontweight="bold",
            color="black",
            linespacing=1.05,
            bbox={
                "boxstyle": "square,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=6,
        )


def main() -> None:
    args = parse_args()
    configure_style()
    values = load_time_statistics(project_path(args.data_root))
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.30, 3.15),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.25},
    )
    draw_count_panel(axes[0], values)
    draw_ratio_panel(axes[1], values)
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.91)
    figure.savefig(output, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
