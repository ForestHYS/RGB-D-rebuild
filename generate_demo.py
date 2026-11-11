from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def intersect_aabb(origin: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    safe = np.where(np.abs(direction) < 1e-9, np.nan, direction)
    first = (lower - origin) / safe
    second = (upper - origin) / safe
    near = np.nanmax(np.minimum(first, second), axis=-1)
    far = np.nanmin(np.maximum(first, second), axis=-1)
    return np.where((far >= near) & (far > 0), np.maximum(near, 0), np.inf)


def render_frame(width: int, height: int, pose_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale_x, scale_y = width / 640.0, height / 480.0
    fx, fy, cx, cy = 517.3 * scale_x, 516.5 * scale_y, 318.6 * scale_x, 255.3 * scale_y
    v, u = np.mgrid[0:height, 0:width]
    camera_rays = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
    world_rays = camera_rays @ pose_c2w[:3, :3].T
    origin = pose_c2w[:3, 3]
    distances = np.full((height, width), np.inf, dtype=np.float64)
    labels = np.zeros((height, width), dtype=np.uint8)

    # Room shell: back, floor, left and right walls.
    planes = [
        (2, 4.0, (-2.4, 2.4), (-1.4, 1.3), 1),
        (1, 1.25, (-2.4, 2.4), (0.2, 4.0), 2),
        (0, -2.2, (-1.4, 1.3), (0.2, 4.0), 3),
        (0, 2.2, (-1.4, 1.3), (0.2, 4.0), 4),
    ]
    for axis, value, range_a, range_b, label in planes:
        denom = world_rays[..., axis]
        t = np.where(np.abs(denom) > 1e-9, (value - origin[axis]) / denom, np.inf)
        hit = origin + world_rays * t[..., None]
        other = [idx for idx in range(3) if idx != axis]
        valid = (
            (t > 0)
            & (hit[..., other[0]] >= range_a[0])
            & (hit[..., other[0]] <= range_a[1])
            & (hit[..., other[1]] >= range_b[0])
            & (hit[..., other[1]] <= range_b[1])
            & (t < distances)
        )
        distances[valid], labels[valid] = t[valid], label

    boxes = [
        (np.array([-0.55, 0.15, 2.0]), np.array([0.35, 1.25, 2.75]), 5),
        (np.array([0.85, 0.60, 2.9]), np.array([1.45, 1.25, 3.5]), 6),
    ]
    for lower, upper, label in boxes:
        t = intersect_aabb(origin, world_rays, lower, upper)
        valid = t < distances
        distances[valid], labels[valid] = t[valid], label

    depth = np.where(np.isfinite(distances), distances * 5000.0, 0).clip(0, 65535).astype(np.uint16)
    palette = np.array(
        [[0, 0, 0], [180, 185, 205], [120, 150, 115], [195, 160, 130], [195, 160, 130], [75, 120, 220], [220, 150, 60]],
        dtype=np.uint8,
    )
    rgb = palette[labels]
    shade = np.clip(0.85 + 0.15 * (u / max(width - 1, 1)), 0, 1)[..., None]
    rgb = (rgb * shade).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth


def _add_ground_truth_points(
    depth: np.ndarray, rgb_bgr: np.ndarray, pose: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    sx, sy = width / 640.0, height / 480.0
    fx, fy, cx, cy = 517.3 * sx, 516.5 * sy, 318.6 * sx, 255.3 * sy
    v, u = np.mgrid[0:height:2, 0:width:2]
    z = depth[::2, ::2].astype(np.float64) / 5000.0
    valid = z > 0.05
    z, u, v = z[valid], u[valid], v[valid]
    camera = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    colors = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[::2, ::2][valid] / 255.0
    return world, colors


def _corrupt_observation(
    rgb: np.ndarray, depth: np.ndarray, index: int, frame_count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    phase = index / max(frame_count - 1, 1)
    # Two deterministic difficult intervals emulate fast motion and sensor dropout.
    severity = max(
        np.exp(-((phase - 0.34) / 0.075) ** 2),
        0.85 * np.exp(-((phase - 0.73) / 0.065) ** 2),
    )
    if severity < 0.08:
        return rgb, depth
    kernel = int(3 + 12 * severity) | 1
    rgb = cv2.GaussianBlur(rgb, (kernel, kernel), 0)
    valid = depth > 0
    # Random holes plus a moving missing-depth stripe.
    drop_probability = 0.12 + 0.60 * severity
    holes = rng.random(depth.shape) < drop_probability
    stripe_width = int(depth.shape[1] * 0.32 * severity)
    stripe_start = int((depth.shape[1] - stripe_width) * phase)
    holes[:, stripe_start : stripe_start + stripe_width] = True
    noisy = depth.astype(np.float32)
    noise = rng.normal(0.0, 18.0 + 65.0 * severity, depth.shape)
    noisy[valid] += noise[valid]
    noisy[holes] = 0
    return rgb, np.clip(noisy, 0, 65535).astype(np.uint16)


def generate(
    root: Path,
    frames: int = 30,
    width: int = 160,
    height: int = 120,
    profile: str = "easy",
    seed: int = 7,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "rgb").mkdir(exist_ok=True)
    (root / "depth").mkdir(exist_ok=True)
    rgb_lines = ["# timestamp filename"]
    depth_lines = ["# timestamp filename"]
    pose_lines = ["# timestamp tx ty tz qx qy qz qw"]
    rng = np.random.default_rng(seed)
    gt_points, gt_colors = [], []
    for index in range(frames):
        phase = index / max(frames - 1, 1)
        if profile == "challenge":
            # Non-uniform motion with a small loop and larger rotations.
            progress = np.clip(phase + 0.075 * np.sin(4 * np.pi * phase), 0.0, 1.0)
            translation = np.array(
                [-0.78 + 1.56 * progress, 0.035 * np.sin(2 * np.pi * phase), 0.10 * np.sin(3 * np.pi * phase)]
            )
            yaw = np.deg2rad(15.0 * np.sin(2.5 * np.pi * phase))
        else:
            translation = np.array([-0.55 + 1.10 * phase, -0.02 + 0.04 * np.sin(phase * np.pi), 0.0])
            yaw = np.deg2rad(8.0 * np.sin((phase - 0.5) * np.pi))
        rotation = Rotation.from_euler("y", yaw)
        pose = np.eye(4)
        pose[:3, :3] = rotation.as_matrix()
        pose[:3, 3] = translation
        timestamp = 1305031102.0 + index / 30.0
        clean_rgb, clean_depth = render_frame(width, height, pose)
        points, colors = _add_ground_truth_points(clean_depth, clean_rgb, pose, width, height)
        gt_points.append(points)
        gt_colors.append(colors)
        if profile == "challenge":
            rgb, depth = _corrupt_observation(clean_rgb, clean_depth, index, frames, rng)
        else:
            rgb, depth = clean_rgb, clean_depth
        rgb_rel = f"rgb/{index:06d}.png"
        depth_rel = f"depth/{index:06d}.png"
        cv2.imwrite(str(root / rgb_rel), rgb)
        cv2.imwrite(str(root / depth_rel), depth)
        quat = rotation.as_quat()
        rgb_lines.append(f"{timestamp:.6f} {rgb_rel}")
        depth_lines.append(f"{timestamp:.6f} {depth_rel}")
        pose_lines.append(
            f"{timestamp:.6f} {translation[0]:.8f} {translation[1]:.8f} {translation[2]:.8f} "
            f"{quat[0]:.8f} {quat[1]:.8f} {quat[2]:.8f} {quat[3]:.8f}"
        )
    (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (root / "depth.txt").write_text("\n".join(depth_lines) + "\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.concatenate(gt_points, axis=0))
    cloud.colors = o3d.utility.Vector3dVector(np.concatenate(gt_colors, axis=0))
    cloud = cloud.voxel_down_sample(0.025)
    o3d.io.write_point_cloud(str(root / "groundtruth.ply"), cloud)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a tiny TUM-style RGB-D demo sequence")
    parser.add_argument("--output", type=Path, default=Path("data/demo_freiburg1"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--profile", choices=["easy", "challenge"], default="easy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    generate(args.output, frames=args.frames, profile=args.profile, seed=args.seed)
    print(f"Demo sequence written to {args.output.resolve()}")
