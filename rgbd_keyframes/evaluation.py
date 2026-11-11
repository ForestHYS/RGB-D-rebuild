from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import open3d as o3d


@dataclass
class QualityMetrics:
    accuracy_mean_m: float
    completeness_mean_m: float
    chamfer_m: float
    fscore_5cm: float


def _downsample(cloud: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if len(cloud.points) == 0:
        return cloud
    return cloud.voxel_down_sample(voxel)


def compare_point_clouds(
    candidate: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    voxel: float = 0.03,
    threshold: float = 0.05,
) -> QualityMetrics:
    candidate = _downsample(candidate, voxel)
    reference = _downsample(reference, voxel)
    if len(candidate.points) == 0 or len(reference.points) == 0:
        return QualityMetrics(float("inf"), float("inf"), float("inf"), 0.0)
    cand_to_ref = np.asarray(candidate.compute_point_cloud_distance(reference))
    ref_to_cand = np.asarray(reference.compute_point_cloud_distance(candidate))
    precision = float(np.mean(cand_to_ref < threshold))
    recall = float(np.mean(ref_to_cand < threshold))
    fscore = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = float(np.mean(cand_to_ref))
    completeness = float(np.mean(ref_to_cand))
    return QualityMetrics(accuracy, completeness, 0.5 * (accuracy + completeness), fscore)


def metrics_dict(metrics: QualityMetrics) -> Dict[str, float]:
    return asdict(metrics)
