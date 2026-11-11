# 资源感知的 RGB-D 关键帧选择与三维重建

本项目从带真值位姿的 RGB-D 视频中选择少量关键帧，并通过 TSDF 融合重建场景。它比较全帧、固定间隔、运动阈值和多指标自适应四种策略，输出三维结果以及质量—资源消耗评测。

## 已实现功能

- 读取和时间同步 TUM RGB-D 的 `rgb.txt`、`depth.txt`、`groundtruth.txt`。
- 根据 Freiburg 1/2/3 自动选择并缩放相机内参。
- 全帧、固定间隔、位姿运动和自适应多指标四种策略。
- 自适应指标：清晰度、有效深度比例、相机平移/旋转、深度一致三维重叠率。
- 帧数预算、最小/最大选帧间隔和图像质量门控。
- Open3D Scalable TSDF 融合、三角网格和点云导出。
- 双向点云距离、Chamfer 风格距离、5 cm 几何 F-score、时间和内存评测。
- 自动生成 CSV、JSON、点云预览和对比图。
- 不依赖外部数据的合成 RGB-D 演示与 pytest 测试。

## 安装

建议使用 Python 3.9～3.11：

```powershell
python -m pip install -r requirements.txt
```

## 立即运行演示

```powershell
python generate_demo.py --output data/demo_freiburg1 --frames 30
python main.py --data data/demo_freiburg1 --output output/demo --max-frames 30 --budget-ratio 0.30 --voxel-length 0.04 --sdf-trunc 0.12
```

仓库中已经保存了一次验证结果：`output/demo/summary.png`。在该合成序列上，自适应方法使用 9/30 帧，5 cm F-score 为 0.996；这个结果只用于验证程序，不应作为最终实验结论。

## 在 TUM RGB-D 上运行

1. 从 TUM RGB-D 官方页面下载一个序列并解压。建议先使用体积较小且运动稳定的 `freiburg1_xyz` 或 `freiburg1_desk`。
2. 确认序列目录下存在 `rgb.txt`、`depth.txt`、`groundtruth.txt`、`rgb/` 和 `depth/`。
3. 运行：

```powershell
python main.py `
  --data data/rgbd_dataset_freiburg1_xyz `
  --output output/freiburg1_xyz `
  --max-frames 120 `
  --input-stride 1 `
  --budget-ratio 0.30
```

首次实验建议限制为 120 帧。确认流程正常后，可提高 `--max-frames`。全帧 TSDF 会作为参考，因此实际会额外执行一次全帧重建。

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--budget-ratio` | 0.30 | 非全帧方法的最大帧数比例 |
| `--uniform-stride` | 4 | 兼容参数；均匀基线当前按预算等距取帧 |
| `--translation-threshold` | 0.08 m | 运动方法的平移阈值 |
| `--rotation-threshold` | 8° | 运动方法的旋转阈值 |
| `--weight-motion` | 0.40 | 自适应策略的运动权重 |
| `--weight-novelty` | 0.40 | 自适应策略的新视野权重 |
| `--weight-quality` | 0.20 | 自适应策略的图像质量权重 |
| `--score-threshold` | 0.48 | 自适应选帧阈值 |
| `--voxel-length` | 0.025 m | TSDF 体素尺寸；增大可降低资源消耗 |
| `--sdf-trunc` | 0.075 m | TSDF 截断距离 |
| `--depth-trunc` | 4.0 m | 最大有效深度 |
| `--methods` | 全部四种 | 可指定部分方法，但程序仍会建立全帧参考 |

三个权重会在程序中自动归一化。其他默认参数位于 `rgbd_keyframes/selection.py` 的 `SelectionConfig`。

## 输出结构

```text
output/<experiment>/
├── run_config.json
├── metrics.csv
├── metrics.json
├── summary.png
├── all/
├── uniform/
├── motion/
└── adaptive/
    ├── mesh.ply
    ├── point_cloud.ply
    ├── preview.png
    └── selection.json
```

`selection.json` 保存最终帧索引、时间戳和各帧的分项得分，可直接用于消融分析和绘图。

## 挑战序列与完整实验

挑战模式包含非匀速相机运动、两段运动模糊、深度空洞和深度噪声，同时生成独立的干净 `groundtruth.ply`，因此评测不会把全帧重建本身当作绝对真值：

```powershell
python generate_demo.py --output data/challenge_freiburg1 --frames 60 --profile challenge --seed 7
python run_experiments.py --data data/challenge_freiburg1 --output output/experiments
```

实验脚本自动完成 10%、15%、20%、30%、40% 五档帧预算，以及去除质量、新颖度或运动项的消融实验，输出 `budget_sweep.png`、`ablation.png` 和对应 CSV/JSON 数据。

本次验证中，自适应方法在 10% 和 15% 预算下分别获得 0.9742 和 0.9798 的 F-score，明显高于均匀抽帧的 0.9480 和 0.9629。完整分析见 `EXPERIMENTS.md`。

## 推荐实验设计

### 主实验

- 数据：`freiburg1_xyz`、`freiburg1_desk`、`freiburg1_room`，每个序列截取相同数量的帧。
- 预算比例：10%、20%、30%、40%、50%。
- 方法：uniform、motion、adaptive。
- 报告：帧数、运行时间、Chamfer 风格距离、F-score 和峰值/增量内存。

### 消融实验

分别通过 `--weight-motion 0`、`--weight-novelty 0` 或 `--weight-quality 0` 移除一个指标；程序会自动归一化剩余权重。也可以用 `--min-blur` 和 `--min-valid-depth` 比较是否启用质量门控。

### 应注意的限制

- 全帧重建只是代理参考，不等同于激光扫描真值。
- 当前实现使用数据集真值位姿，没有评估 SLAM 轨迹误差。
- 内存指标是进程 RSS 增量，容易受到 Python/Open3D 内存复用影响；正式报告最好同时记录操作系统监控值。
- TUM 的部分序列存在深度空洞和运动模糊，应报告所用序列、帧范围和参数。

## 测试

```powershell
python -m pytest -q
```

测试会生成一个小型 TUM 格式合成场景，验证数据读取、选帧、TSDF 重建和质量评测的完整流程。
