from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RGBDFrame:
    timestamp: float
    rgb_path: Path
    depth_path: Path
    pose_c2w: np.ndarray


TUM_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def _read_tum_list(path: Path) -> List[Tuple[float, List[str]]]:
    rows: List[Tuple[float, List[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append((float(parts[0]), parts[1:]))
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _nearest(
    rows: Sequence[Tuple[float, List[str]]], timestamp: float, max_delta: float
) -> Tuple[float, List[str]]:
    times = [row[0] for row in rows]
    pos = bisect_left(times, timestamp)
    choices = []
    if pos < len(rows):
        choices.append(rows[pos])
    if pos > 0:
        choices.append(rows[pos - 1])
    match = min(choices, key=lambda row: abs(row[0] - timestamp))
    if abs(match[0] - timestamp) > max_delta:
        raise LookupError
    return match


def pose_from_tum(values: Iterable[str]) -> np.ndarray:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size != 7:
        raise ValueError("A TUM pose must contain tx ty tz qx qy qz qw")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(vals[3:]).as_matrix()
    pose[:3, 3] = vals[:3]
    return pose


def load_tum_sequence(root: Path, max_time_delta: float = 0.03) -> List[RGBDFrame]:
    """Load synchronized RGB, depth and ground-truth pose records."""
    root = Path(root)
    required = [root / "rgb.txt", root / "depth.txt", root / "groundtruth.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing TUM files: " + ", ".join(missing))

    rgb_rows = _read_tum_list(required[0])
    depth_rows = _read_tum_list(required[1])
    pose_rows = _read_tum_list(required[2])
    frames: List[RGBDFrame] = []
    for rgb_time, rgb_values in rgb_rows:
        try:
            _, depth_values = _nearest(depth_rows, rgb_time, max_time_delta)
            pose_time, pose_values = _nearest(pose_rows, rgb_time, max_time_delta)
        except LookupError:
            continue
        frames.append(
            RGBDFrame(
                timestamp=pose_time,
                rgb_path=root / rgb_values[0],
                depth_path=root / depth_values[0],
                pose_c2w=pose_from_tum(pose_values),
            )
        )
    if not frames:
        raise ValueError("No RGB/depth/pose triplets could be synchronized")
    return frames


def infer_intrinsics(root: Path, width: int, height: int) -> CameraIntrinsics:
    name = str(root).lower()
    values = TUM_INTRINSICS["freiburg1"]
    for key, candidate in TUM_INTRINSICS.items():
        if key in name:
            values = candidate
            break
    fx, fy, cx, cy = values
    # TUM calibration targets 640x480. Scale if images were resized.
    sx, sy = width / 640.0, height / 480.0
    return CameraIntrinsics(width, height, fx * sx, fy * sy, cx * sx, cy * sy)


def inspect_sequence(root: Path, frames: Sequence[RGBDFrame]) -> CameraIntrinsics:
    color = cv2.imread(str(frames[0].rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(frames[0].depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise FileNotFoundError("Could not read the first RGB-D frame")
    if color.shape[:2] != depth.shape[:2]:
        raise ValueError("RGB and depth dimensions do not match")
    height, width = depth.shape[:2]
    return infer_intrinsics(root, width, height)
