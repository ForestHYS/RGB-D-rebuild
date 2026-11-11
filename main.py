from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import open3d as o3d

from rgbd_keyframes.dataset import inspect_sequence, load_tum_sequence
from rgbd_keyframes.evaluation import compare_point_clouds, metrics_dict
from rgbd_keyframes.reconstruction import reconstruct_tsdf, save_reconstruction
from rgbd_keyframes.selection import (
    SelectionConfig,
    scores_to_dicts,
    select_adaptive,
    select_all,
    select_motion,
    select_uniform,
)
from rgbd_keyframes.visualization import save_point_cloud_preview, save_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resource-aware keyframe selection and RGB-D TSDF reconstruction"
    )
    parser.add_argument("--data", type=Path, required=True, help="TUM RGB-D sequence directory")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--methods", nargs="+", choices=["all", "uniform", "motion", "adaptive"],
        default=["all", "uniform", "motion", "adaptive"],
    )
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--input-stride", type=int, default=1)
    parser.add_argument("--budget-ratio", type=float, default=0.30)
    parser.add_argument("--uniform-stride", type=int, default=4)
    parser.add_argument("--translation-threshold", type=float, default=0.08)
    parser.add_argument("--rotation-threshold", type=float, default=8.0)
    parser.add_argument("--weight-motion", type=float, default=0.40)
    parser.add_argument("--weight-novelty", type=float, default=0.40)
    parser.add_argument("--weight-quality", type=float, default=0.20)
    parser.add_argument("--score-threshold", type=float, default=0.48)
    parser.add_argument("--min-blur", type=float, default=10.0)
    parser.add_argument("--min-valid-depth", type=float, default=0.25)
    parser.add_argument("--sample-step", type=int, default=12)
    parser.add_argument("--voxel-length", type=float, default=0.025)
    parser.add_argument("--sdf-trunc", type=float, default=0.075)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.budget_ratio <= 1):
        raise ValueError("--budget-ratio must be in (0, 1]")
    weight_sum = args.weight_motion + args.weight_novelty + args.weight_quality
    if weight_sum <= 0 or min(args.weight_motion, args.weight_novelty, args.weight_quality) < 0:
        raise ValueError("Selection weights must be non-negative and have a positive sum")
    frames = load_tum_sequence(args.data)
    frames = frames[:: max(1, args.input_stride)][: args.max_frames]
    if len(frames) < 2:
        raise ValueError("At least two synchronized frames are required")
    intr = inspect_sequence(args.data, frames)
    args.output.mkdir(parents=True, exist_ok=True)
    config = SelectionConfig(
        budget_ratio=args.budget_ratio,
        uniform_stride=args.uniform_stride,
        translation_threshold=args.translation_threshold,
        rotation_threshold_deg=args.rotation_threshold,
        weight_motion=args.weight_motion / weight_sum,
        weight_novelty=args.weight_novelty / weight_sum,
        weight_quality=args.weight_quality / weight_sum,
        score_threshold=args.score_threshold,
        min_blur=args.min_blur,
        min_valid_depth=args.min_valid_depth,
        sample_step=max(1, args.sample_step),
    )
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "available_frames": len(frames),
                "intrinsics": asdict(intr),
                "selection": asdict(config),
                "voxel_length": args.voxel_length,
                "sdf_trunc": args.sdf_trunc,
                "depth_trunc": args.depth_trunc,
                "reference": "groundtruth.ply" if (args.data / "groundtruth.ply").exists() else "all_frames",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    selectors: Dict[str, Callable] = {
        "all": lambda fs, cfg: select_all(fs, cfg),
        "uniform": lambda fs, cfg: select_uniform(fs, cfg),
        "motion": lambda fs, cfg: select_motion(fs, cfg),
        "adaptive": lambda fs, cfg: select_adaptive(fs, intr, cfg),
    }
    requested = list(dict.fromkeys(args.methods))
    # A full-frame reference is always needed for quality evaluation.
    execution_order = ["all"] + [method for method in requested if method != "all"]
    groundtruth_path = args.data / "groundtruth.ply"
    reference_cloud = o3d.io.read_point_cloud(str(groundtruth_path)) if groundtruth_path.exists() else None
    if reference_cloud is not None and len(reference_cloud.points) == 0:
        reference_cloud = None
    reference_source = "external_groundtruth" if reference_cloud is not None else "all_frames"
    rows: List[dict] = []
    for method in execution_order:
        indices, score_records = selectors[method](frames, config)
        method_dir = args.output / method
        result = reconstruct_tsdf(
            frames,
            indices,
            intr,
            voxel_length=args.voxel_length,
            sdf_trunc=args.sdf_trunc,
            depth_trunc=args.depth_trunc,
        )
        save_reconstruction(result, method_dir)
        save_point_cloud_preview(result.point_cloud, method_dir / "preview.png")
        if method == "all" and reference_cloud is None:
            reference_cloud = result.point_cloud
        quality = compare_point_clouds(result.point_cloud, reference_cloud)
        row = {
            "method": method,
            "selected_frames": len(indices),
            "frame_ratio": len(indices) / len(frames),
            "runtime_seconds": result.elapsed_seconds,
            "memory_delta_mb": result.memory_delta_mb,
            "reference_source": reference_source,
            **metrics_dict(quality),
        }
        rows.append(row)
        (method_dir / "selection.json").write_text(
            json.dumps(
                {"indices": indices, "timestamps": [frames[i].timestamp for i in indices], "scores": scores_to_dicts(score_records)},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"{method:>8}: {len(indices):4d}/{len(frames)} frames, "
            f"{result.elapsed_seconds:7.2f}s, F-score={quality.fscore_5cm:.3f}"
        )

    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_summary(rows, args.output / "summary.png")
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
