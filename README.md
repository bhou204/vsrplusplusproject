# 基于运动强度 ROI 的区域增强视频超分原型

这是一个最小可运行的区域感知视频超分原型，目标是在尽量少改动官方 BasicVSR++ 仓库的前提下，先跑通一个可比较的实验框架：

- `full_heavy`：整段视频都交给 BasicVSR++
- `roi_heavy`：背景先做 bicubic，运动显著 ROI 再交给 BasicVSR++，最后拼回整帧

当前版本的核心价值不是引入新模型，而是把“ROI 感知推理”这条工程链路搭起来，便于后续替换 ROI 方式或轻量增强模块。

## 1. 项目简介

本项目以 `BasicVSR_PlusPlus/` 作为重模型引擎，通过 `roi_vsr_project/` 中的外层 pipeline 完成：

1. 读取输入视频或帧序列
2. 基于相邻帧差分估计运动强度
3. 自动得到稳定 ROI bbox
4. 非 ROI 区域先用 bicubic 上采样
5. ROI 区域单独裁剪后交给 BasicVSR++
6. 将 ROI 结果拼回整帧
7. 输出结果并统计时间、FPS 和 GPU 显存峰值

## 2. 方法说明

这里使用“运动强度”做 ROI，是为了在不引入额外检测模型的前提下尽快得到一个可跑通的区域感知原型。思路很直接：

- 相邻帧之间运动越强，越可能存在细节退化更明显或值得重建的区域
- 用灰度差分、阈值和形态学操作，就能得到一个足够轻量的运动区域估计
- 对多帧 ROI 做简单平滑，可以减少抖动，避免区域在时间上剧烈跳变

当前版本优先实现 bbox 方案，因为它最容易和 BasicVSR++ 的帧序列推理衔接。mask 和 feather blending 已保留接口，后续可以继续增强。

## 3. 目录结构

```text
roi_vsr_project/
├── configs/
│   └── default.yaml          # 默认配置，集中管理路径和实验参数
├── data/                     # 输入数据目录，建议放样例视频或帧序列
├── results/                  # 输出目录，保存增强结果和中间产物
├── scripts/                  # 便捷运行脚本
├── src/                      # 核心代码
├── main.py                   # 主入口，串联完整 pipeline
└── README.md                 # 中文说明文档
```

### 各模块用途

- `src/io_utils.py`：视频读取、拆帧、保存帧、写视频
- `src/roi_motion.py`：基于帧差分的运动 ROI 估计
- `src/light_enhance.py`：bicubic 轻量上采样
- `src/heavy_bvsr.py`：通过子进程调用官方 BasicVSR++ 推理脚本
- `src/fusion.py`：ROI 拼回整帧和可选 feather blending
- `src/benchmark.py`：统计耗时、FPS 和 GPU 显存峰值
- `main.py`：命令行入口，支持 `full_heavy`、`roi_heavy` 和 `compare`

## 4. 环境说明

假设你的 BasicVSR++ 环境已经配置完成，且以下条件满足：

- `BasicVSR_PlusPlus/` 仓库在本项目同级目录
- `BasicVSR_PlusPlus/chkpts/basicvsr_plusplus_reds4.pth` 存在
- Python 环境中已安装 BasicVSR++ 所需依赖
- 能够正常执行官方仓库中的 `demo/restoration_video_demo.py`

本项目本身只使用轻量依赖：`os`、`pathlib`、`cv2`、`numpy`、`subprocess`、`time`、`yaml/json` 等。

## 5. 如何准备输入数据

支持两种输入形式：

1. 视频文件，例如 `data/demo_000/demo.mp4`
2. 帧序列目录，例如官方仓库自带的 `../BasicVSR_PlusPlus/data/demo_000/`

现在如果你把多个视频直接放到 `roi_vsr_project/data/` 下，例如 `grass.mp4` 和 `road.mp4`，程序会在运行时自动按文件逐个处理，不需要你手工一个个启动。

## 6. 如何配置路径

默认配置文件是 `configs/default.yaml`。你只需要重点检查这些字段：

- `input_video_dir`
- `basicvsr_root`
- `basicvsr_config`
- `basicvsr_checkpoint`
- `input_video_path`
- `input_frames_dir`
- `temp_dir`
- `output_dir`
- `video_output_dir`
- `summary_dir`

默认配置已经指向 `data/` 目录，因此只要你把视频放进 `roi_vsr_project/data/`，就会自动批处理。

如果你还想处理官方仓库自带的帧序列，也可以把 `input_video_dir` 置空，再改回 `input_frames_dir`。

## 7. 如何运行 full_heavy

### 方式一：直接运行主入口

```bash
python main.py --config configs/default.yaml
```

### 方式二：使用便捷脚本

```bash
python scripts/run_compare.py
```

现在主入口默认会对 `data/` 下的每个视频同时跑 `full_heavy` 和 `roi_heavy`，并自动输出视频、帧和摘要结果。

## 8. 如何运行 roi_heavy

### 方式一：直接运行主入口

```bash
python main.py --config configs/default.yaml
```

### 方式二：使用便捷脚本

```bash
python scripts/run_compare.py
```

当前批处理流程会自动同时生成两种结果：

1. 读取原始低分辨率帧
2. 计算运动 ROI bbox
3. 全图先 bicubic 上采样
4. ROI 裁剪后送入 BasicVSR++
5. 将 ROI 增强结果粘回整帧

输出会按视频名分别写入 `results/output/<video_name>/roi_heavy/`。

## 9. 输出结果说明

运行完成后，默认会得到：

- `results/output/<video_name>/full_heavy/`：每个视频各自的 full_heavy 输出帧
- `results/output/<video_name>/roi_heavy/`：每个视频各自的 ROI-heavy 输出帧
- `results/output/<video_name>/original_frames/`：原始输入帧的保存副本，若开启 `save_intermediate`
- `results/video/<video_name>/`：自动导出的 mp4 视频，包括原始输入、full_heavy、roi_heavy
- `results/output/<video_name>/summary/metrics.json`：每个视频的耗时、FPS、PSNR、SSIM、ROI 面积占比
- `results/summary/run_summary_*.json`：本次批处理的总汇总

如果你开启了 `save_intermediate: true`，中间结果会更完整，便于调试 ROI 是否稳定、裁剪是否正确、融合是否对齐。

## 10. benchmark 指标说明

当前版本会尽量输出以下指标：

- 总耗时 `total_seconds`
- 平均每帧耗时 `avg_frame_seconds`
- FPS `fps`
- GPU 显存峰值 `peak_gpu_memory_mib`，通过 `nvidia-smi` 轮询获取，属于 best-effort
- ROI 平均面积占比 `average_roi_area_ratio`，只在 `roi_heavy` 中有效，表示被认为需要重点增强的区域占整帧面积的比例
- `PSNR` / `SSIM`：当前实现中按 full_heavy 作为参考，和 roi_heavy 输出逐帧对比，属于相对一致性指标，不是带 GT 的绝对质量指标
- 输出目录 `output_dir`

如果 `nvidia-smi` 不可用，显存峰值会显示为 `N/A`。

如果你的视频比较长，默认配置已经把 `max_seq_len` 设成了一个较小值，用来避免 BasicVSR++ 一次性把整段序列送进 GPU 导致显存爆掉。你也可以按你的显卡情况再调小或调大这个值。

## 11. 当前版本局限性

- ROI 估计只用帧差分，没有引入目标检测或分割模型
- 当前主流程优先实现 bbox 方案，mask 级融合还只是预留接口
- ROI 采用全局稳定框，适合原型验证，不适合复杂多目标场景的精细重建
- `roi_heavy` 中重模型只作用于 ROI crop，crop 尺寸和边界处理还比较朴素
- GPU 显存统计是 best-effort，跨子进程只能粗略采样
- 当前 PSNR / SSIM 不是对 GT 的质量评估，而是 full_heavy 和 roi_heavy 的相对比较

## 12. 后续扩展方向

这个原型后续比较自然的扩展方向有：

- 把 ROI 生成从差分法替换成 detector / segmenter
- 将 `bicubic` 背景替换为轻量版 BasicVSR++ 或其他轻模型
- 增加更稳定的时序 ROI 跟踪和框平滑
- 增加 mask 级别融合与边缘 feather blending
- 支持多 ROI、分块推理和自适应 ROI 扩展
- 增加对视频输入和帧输入的统一输出封装
- 如果后续有 GT，可以把 PSNR / SSIM 的参考从 full_heavy 切换成真实高质量标注

## 13. 运行建议

建议先把 `data/` 下的两个视频放好，直接运行：

```bash
python main.py --config configs/default.yaml
```

这样可以一次性得到 `full_heavy` 和 `roi_heavy` 的对比结果，并自动导出视频、帧和统计摘要。
