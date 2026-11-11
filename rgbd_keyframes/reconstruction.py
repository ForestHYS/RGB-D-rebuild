from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import cv2
import numpy as np
import open3d as o3d
import psutil

from .dataset import CameraIntrinsics, RGBDFrame


@dataclass
class ReconstructionResult:
    mesh: o3d.geometry.TriangleMesh
    point_cloud: o3d.geometry.PointCloud
    elapsed_seconds: float
    memory_delta_mb: float


def to_open3d_intrinsics(intr: CameraIntrinsics) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
    )


def reconstruct_tsdf(
    frames: Sequence[RGBDFrame],
    indices: Sequence[int],
    intr: CameraIntrinsics,
    voxel_length: float = 0.02,
    sdf_trunc: float = 0.06,
    depth_trunc: float = 4.0,
) -> ReconstructionResult:
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024**2
    start = perf_counter()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    camera = to_open3d_intrinsics(intr)
    for idx in indices:
        frame = frames[idx]
        color_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_raw is None:
            raise FileNotFoundError(f"Unable to read RGB-D frame {idx}")
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color_rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth_raw)),
            depth_scale=5000.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        world_to_camera = np.linalg.inv(frame.pose_c2w)
        volume.integrate(rgbd, camera, world_to_camera)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    point_cloud = volume.extract_point_cloud()
    elapsed = perf_counter() - start
    memory_after = process.memory_info().rss / 1024**2
    return ReconstructionResult(mesh, point_cloud, elapsed, max(0.0, memory_after - memory_before))


def save_reconstruction(result: ReconstructionResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_dir / "mesh.ply"), result.mesh)
    o3d.io.write_point_cloud(str(output_dir / "point_cloud.ply"), result.point_cloud)
