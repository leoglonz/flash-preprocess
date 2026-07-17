"""Stacked bar chart of storm-event train/test counts per gauge.

Global date-window split: train 2020/01/01-2024/09/30, test 2024/10/01-2025/12/31
(see README.md). Run with:
    python plot/train_test_split.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA = [
    ("02085000", 408, 136),
    ("02085070", 389, 111),
    ("02085500", 221, 43),
    ("02086500", 471, 137),
    ("02086624", 340, 86),
    ("02086849", 519, 190),
    ("02087275", 293, 63),
    ("02087324", 298, 83),
    ("02087359", 340, 65),
    ("02087580", 435, 79),
    ("02088000", 329, 102),
    ("02088500", 212, 37),
]

TRAIN_COLOR = "#2a78d6"
TEST_COLOR = "#1baf7a"
OUT_PATH = Path(__file__).parent / "train_test_split.png"


def main() -> None:
    rows = sorted(DATA, key=lambda r: r[1] + r[2], reverse=True)
    gauges = [r[0] for r in rows]
    train = np.array([r[1] for r in rows])
    test = np.array([r[2] for r in rows])
    total = train + test

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    x = np.arange(len(gauges))
    width = 0.6

    ax.bar(x, train, width, label="Train (2020-2024)", color=TRAIN_COLOR,
           edgecolor="white", linewidth=0.6)
    ax.bar(x, test, width, bottom=train, label="Test (2024-2025)", color=TEST_COLOR,
           edgecolor="white", linewidth=0.6)

    for xi, t in zip(x, total):
        ax.text(xi, t + 8, f"{t:,}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#0b0b0b")

    ax.set_xticks(x)
    ax.set_xticklabels(gauges, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Flash Flood Events", fontsize=11)
    ax.set_title(
        "Flash Flood Event Data Split, by USGS Gauge\n"
        "Train [2020/01/01 - 2024/09/30]  |  Test [2024/10/01 - 2025/12/31]",
        fontsize=13, loc="left",
    )
    ax.set_ylim(0, total.max() * 1.15)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    n_train, n_test = train.sum(), test.sum()
    ax.text(
        0.99, -0.30,
        f"Total: {n_train + n_test:,} events   "
        f"({n_train:,} train, {n_test:,} test, {n_test / (n_train + n_test):.1%} test share)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#52514e",
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
