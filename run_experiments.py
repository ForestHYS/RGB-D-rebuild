from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_main(data: Path, output: Path, budget: float, extra: list[str] | None = None) -> list[dict]:
    command = [
        sys.executable, "main.py", "--data", str(data), "--output", str(output),
        "--max-frames", "60", "--budget-ratio", str(budget),
        "--voxel-length", "0.04", "--sdf-trunc", "0.12",
    ]
    if extra:
        command.extend(extra)
    subprocess.run(command, check=True)
    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))


def save_sweep_plot(rows: list[dict], path: Path) -> None:
    methods = ["uniform", "motion", "adaptive"]
    colors = {"uniform": "#4C78A8", "motion": "#F58518", "adaptive": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
    for method in methods:
        values = sorted((row for row in rows if row["method"] == method), key=lambda row: row["budget"])
        x = [100 * row["budget"] for row in values]
        axes[0].plot(x, [row["fscore_5cm"] for row in values], "o-", label=method, color=colors[method])
        axes[1].plot(x, [1000 * row["chamfer_m"] for row in values], "o-", label=method, color=colors[method])
        axes[2].plot(x, [row["runtime_seconds"] for row in values], "o-", label=method, color=colors[method])
    axes[0].set_ylabel("F-score @ 5 cm")
    axes[1].set_ylabel("Mean bidirectional error (mm)")
    axes[2].set_ylabel("TSDF runtime (s)")
    for axis in axes:
        axis.set_xlabel("Frame budget (%)")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Quality-resource trade-off on the challenge sequence")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_ablation_plot(rows: list[dict], path: Path) -> None:
    labels = [row["variant"] for row in rows]
    scores = [row["fscore_5cm"] for row in rows]
    fig, axis = plt.subplots(figsize=(7.5, 4), dpi=150)
    bars = axis.bar(labels, scores, color=["#54A24B", "#E45756", "#B279A2", "#FF9DA6"])
    axis.set_ylim(min(scores) - 0.01, max(scores) + 0.004)
    axis.set_ylabel("F-score @ 5 cm")
    axis.set_title("Adaptive selector ablation at 15% frame budget")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, scores):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.0005, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run budget sweep and adaptive-selector ablations")
    parser.add_argument("--data", type=Path, default=Path("data/challenge_freiburg1"))
    parser.add_argument("--output", type=Path, default=Path("output/experiments"))
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.30, 0.40])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    for budget in args.budgets:
        metrics = run_main(args.data, args.output / f"budget_{int(100 * budget):02d}", budget)
        for row in metrics:
            if row["method"] != "all":
                sweep_rows.append({"budget": budget, **row})
    (args.output / "sweep_metrics.json").write_text(json.dumps(sweep_rows, indent=2), encoding="utf-8")
    with (args.output / "sweep_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    save_sweep_plot(sweep_rows, args.output / "budget_sweep.png")

    variants = {
        "full": [],
        "no quality": ["--weight-quality", "0", "--min-blur", "0", "--min-valid-depth", "0"],
        "no novelty": ["--weight-novelty", "0"],
        "no motion": ["--weight-motion", "0"],
    }
    ablation_rows = []
    for name, extra in variants.items():
        metrics = run_main(args.data, args.output / f"ablation_{name.replace(' ', '_')}", 0.15, extra)
        adaptive = next(row for row in metrics if row["method"] == "adaptive")
        ablation_rows.append({"variant": name, **adaptive})
    (args.output / "ablation_metrics.json").write_text(json.dumps(ablation_rows, indent=2), encoding="utf-8")
    save_ablation_plot(ablation_rows, args.output / "ablation.png")
    print(f"Experiment report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
