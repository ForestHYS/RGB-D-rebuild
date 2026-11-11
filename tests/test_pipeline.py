from pathlib import Path

import numpy as np

from generate_demo import generate
from rgbd_keyframes.dataset import inspect_sequence, load_tum_sequence, pose_from_tum
from rgbd_keyframes.evaluation import compare_point_clouds
from rgbd_keyframes.reconstruction import reconstruct_tsdf
from rgbd_keyframes.selection import SelectionConfig, pose_delta, select_adaptive, select_uniform


def test_pose_and_delta():
    pose = pose_from_tum(["1", "2", "3", "0", "0", "0", "1"])
    np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])
    translation, rotation = pose_delta(np.eye(4), pose)
    assert np.isclose(translation, np.sqrt(14))
    assert np.isclose(rotation, 0)


def test_demo_selection_and_reconstruction(tmp_path: Path):
    root = tmp_path / "demo_freiburg1"
    generate(root, frames=10, width=80, height=60)
    assert (root / "groundtruth.ply").exists()
    frames = load_tum_sequence(root)
    intr = inspect_sequence(root, frames)
    config = SelectionConfig(budget_ratio=0.4, uniform_stride=3, min_gap=1, max_gap=4)
    uniform, _ = select_uniform(frames, config)
    adaptive, scores = select_adaptive(frames, intr, config)
    assert uniform[0] == adaptive[0] == 0
    assert uniform[-1] == adaptive[-1] == len(frames) - 1
    assert len(adaptive) <= 4
    assert len(scores) == len(frames)

    reference = reconstruct_tsdf(frames, list(range(len(frames))), intr, voxel_length=0.05, sdf_trunc=0.15)
    candidate = reconstruct_tsdf(frames, adaptive, intr, voxel_length=0.05, sdf_trunc=0.15)
    quality = compare_point_clouds(candidate.point_cloud, reference.point_cloud, voxel=0.05)
    assert len(reference.point_cloud.points) > 0
    assert quality.fscore_5cm > 0.25
