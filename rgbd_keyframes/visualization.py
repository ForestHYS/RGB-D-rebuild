from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def save_point_cloud_preview(cloud: o3d.geometry.PointCloud, path: Path, max_points: int = 30000) -> None:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if len(points) == 0:
        return
    if len(points) > max_points:
        rng = np.random.default_rng(7)
        choice = rng.choice(len(points), max_points, replace=False)
        points, colors = points[choice], colors[choice]
    fig = plt.figure(figsize=(8, 6), dpi=140)
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 2], -points[:, 1], s=0.3, c=colors if len(colors) else None)
    axis.set_title("Reconstructed point cloud")
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_zlabel("-Y")
    axis.view_init(elev=24, azim=-62)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_summary(metrics: Sequence[Dict], path: Path) -> None:
    methods = [row["method"] for row in metrics]
    frames = [row["selected_frames"] for row in metrics]
    times = [row["runtime_seconds"] for row in metrics]
    fscores = [row["fscore_5cm"] for row in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), dpi=140)
    axes[0].bar(methods, frames, color="#4C78A8")
    axes[0].set_title("Selected frames")
    axes[1].bar(methods, times, color="#F58518")
    axes[1].set_title("Runtime (s)")
    axes[2].bar(methods, fscores, color="#54A24B")
    axes[2].set_title("F-score @ 5 cm")
    axes[2].set_ylim(0, 1.05)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
