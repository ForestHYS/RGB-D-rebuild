from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .dataset import CameraIntrinsics, RGBDFrame


@dataclass
class SelectionConfig:
    budget_ratio: float = 0.30
    uniform_stride: int = 4
    translation_threshold: float = 0.08
    rotation_threshold_deg: float = 8.0
    blur_target: float = 100.0
    min_blur: float = 10.0
    min_valid_depth: float = 0.25
    overlap_depth_tolerance: float = 0.08
    sample_step: int = 12
    min_gap: int = 2
    max_gap: int = 12
    score_threshold: float = 0.48
    weight_motion: float = 0.40
    weight_novelty: float = 0.40
    weight_quality: float = 0.20


@dataclass
class FrameScore:
    index: int
    selected: bool
    score: float
    translation: float
    rotation_deg: float
    overlap: float
    blur: float
    valid_depth: float


def pose_delta(first: np.ndarray, second: np.ndarray) -> Tuple[float, float]:
    relative = np.linalg.inv(first) @ second
    translation = float(np.linalg.norm(relative[:3, 3]))
    rotation = float(np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude()))
    return translation, rotation


def _read_gray_depth(frame: RGBDFrame) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.imread(str(frame.rgb_path), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
    if gray is None or depth is None:
        raise FileNotFoundError(f"Unable to read frame at {frame.rgb_path}")
    return gray, depth.astype(np.float32) / 5000.0


def frame_quality(frame: RGBDFrame, depth_trunc: float = 4.0) -> Tuple[float, float]:
    gray, depth = _read_gray_depth(frame)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    valid = float(np.mean((depth > 0.05) & (depth < depth_trunc)))
    return blur, valid


def estimate_overlap(
    source: RGBDFrame,
    target: RGBDFrame,
    intr: CameraIntrinsics,
    step: int = 12,
    depth_tolerance: float = 0.08,
) -> float:
    """Approximate depth-consistent overlap from source into target."""
    _, source_depth = _read_gray_depth(source)
    _, target_depth = _read_gray_depth(target)
    v, u = np.mgrid[0 : intr.height : step, 0 : intr.width : step]
    z = source_depth[v, u]
    valid = (z > 0.05) & (z < 4.0)
    if valid.sum() < 10:
        return 0.0
    u, v, z = u[valid], v[valid], z[valid]
    points = np.stack(
        ((u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z, np.ones_like(z)),
        axis=0,
    )
    target_from_source = np.linalg.inv(target.pose_c2w) @ source.pose_c2w
    transformed = target_from_source @ points
    positive = transformed[2] > 0.05
    transformed = transformed[:, positive]
    if transformed.shape[1] == 0:
        return 0.0
    px = np.rint(intr.fx * transformed[0] / transformed[2] + intr.cx).astype(int)
    py = np.rint(intr.fy * transformed[1] / transformed[2] + intr.cy).astype(int)
    inside = (px >= 0) & (px < intr.width) & (py >= 0) & (py < intr.height)
    if inside.sum() == 0:
        return 0.0
    observed = target_depth[py[inside], px[inside]]
    predicted = transformed[2, inside]
    consistent = (observed > 0.05) & (np.abs(observed - predicted) < depth_tolerance)
    return float(consistent.sum() / max(len(z), 1))


def _limit_budget(indices: List[int], scores: Dict[int, float], total: int, ratio: float) -> List[int]:
    budget = max(2, min(total, ceil(total * ratio)))
    if len(indices) <= budget:
        return sorted(set(indices))
    # Preserve both endpoints, then greedily balance utility and temporal coverage.
    selected = {0, total - 1}
    candidates = set(indices) - selected
    while candidates and len(selected) < budget:
        def utility(idx: int) -> float:
            nearest = min(abs(idx - chosen) for chosen in selected) / max(total - 1, 1)
            return scores.get(idx, 0.0) * (0.55 + 0.45 * nearest)

        best = max(candidates, key=utility)
        selected.add(best)
        candidates.remove(best)
    return sorted(selected)


def _limit_adaptive_budget(
    indices: List[int], scores: Dict[int, float], frames: Sequence[RGBDFrame], config: SelectionConfig
) -> List[int]:
    """Budgeted selection balancing informativeness and pose-space coverage."""
    total = len(frames)
    budget = max(2, min(total, ceil(total * config.budget_ratio)))
    selected = {0, total - 1}
    candidates = set(indices) - selected
    while candidates and len(selected) < budget:
        def utility(idx: int) -> float:
            pose_distances = []
            for chosen in selected:
                trans, rot = pose_delta(frames[chosen].pose_c2w, frames[idx].pose_c2w)
                pose_distances.append(
                    min(1.0, 0.5 * trans / config.translation_threshold + 0.5 * rot / config.rotation_threshold_deg)
                )
            pose_coverage = min(pose_distances)
            temporal = min(abs(idx - chosen) for chosen in selected) / max(total - 1, 1)
            return 0.55 * scores.get(idx, 0.0) + 0.30 * pose_coverage + 0.15 * temporal

        best = max(candidates, key=utility)
        selected.add(best)
        candidates.remove(best)
    return sorted(selected)


def select_all(frames: Sequence[RGBDFrame], _: SelectionConfig) -> Tuple[List[int], List[FrameScore]]:
    return list(range(len(frames))), []


def select_uniform(frames: Sequence[RGBDFrame], config: SelectionConfig) -> Tuple[List[int], List[FrameScore]]:
    budget = max(2, min(len(frames), ceil(len(frames) * config.budget_ratio)))
    indices = np.rint(np.linspace(0, len(frames) - 1, budget)).astype(int).tolist()
    return sorted(set(indices)), []


def select_motion(frames: Sequence[RGBDFrame], config: SelectionConfig) -> Tuple[List[int], List[FrameScore]]:
    chosen = [0]
    scores = {0: 1.0}
    records: List[FrameScore] = []
    last = 0
    for idx in range(1, len(frames)):
        trans, rot = pose_delta(frames[last].pose_c2w, frames[idx].pose_c2w)
        score = max(trans / config.translation_threshold, rot / config.rotation_threshold_deg)
        take = trans >= config.translation_threshold or rot >= config.rotation_threshold_deg
        records.append(FrameScore(idx, take, score, trans, rot, 1.0, 0.0, 0.0))
        if take:
            chosen.append(idx)
            scores[idx] = score
            last = idx
    if chosen[-1] != len(frames) - 1:
        chosen.append(len(frames) - 1)
        scores[len(frames) - 1] = 1.0
    return _limit_budget(chosen, scores, len(frames), config.budget_ratio), records


def select_adaptive(
    frames: Sequence[RGBDFrame], intr: CameraIntrinsics, config: SelectionConfig
) -> Tuple[List[int], List[FrameScore]]:
    chosen = [0]
    score_map = {0: 1.0, len(frames) - 1: 1.0}
    candidates = [0]
    records: List[FrameScore] = [FrameScore(0, True, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
    last = 0
    for idx in range(1, len(frames)):
        gap = idx - last
        trans, rot = pose_delta(frames[last].pose_c2w, frames[idx].pose_c2w)
        blur, valid = frame_quality(frames[idx])
        overlap = estimate_overlap(
            frames[last], frames[idx], intr, config.sample_step, config.overlap_depth_tolerance
        )
        motion = min(1.0, 0.5 * trans / config.translation_threshold + 0.5 * rot / config.rotation_threshold_deg)
        # Missing/noisy depth can look like novel geometry. Reliability weighting
        # prevents low-validity frames from receiving a falsely high novelty score.
        novelty = (1.0 - np.clip(overlap, 0.0, 1.0)) * min(valid / 0.60, 1.0)
        quality = 0.5 * min(blur / config.blur_target, 1.0) + 0.5 * valid
        score = float(
            config.weight_motion * motion
            + config.weight_novelty * novelty
            + config.weight_quality * quality
        )
        acceptable = blur >= config.min_blur and valid >= config.min_valid_depth
        take = acceptable and gap >= config.min_gap and (
            score >= config.score_threshold or gap >= config.max_gap
        )
        records.append(FrameScore(idx, take, score, trans, rot, overlap, blur, valid))
        if acceptable:
            candidates.append(idx)
            score_map[idx] = score
        if take:
            chosen.append(idx)
            last = idx
    if chosen[-1] != len(frames) - 1:
        chosen.append(len(frames) - 1)
        score_map[len(frames) - 1] = 1.0
    if len(frames) - 1 not in candidates:
        candidates.append(len(frames) - 1)
    limited = _limit_adaptive_budget(candidates, score_map, frames, config)
    selected_set = set(limited)
    for record in records:
        record.selected = record.index in selected_set
    return limited, records


def scores_to_dicts(scores: Sequence[FrameScore]) -> List[dict]:
    return [asdict(score) for score in scores]
