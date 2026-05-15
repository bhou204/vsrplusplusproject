# 像素级不确定性感知混合视频超分 (Pixel-level Uncertainty-aware Hybrid VSR)

## 📋 项目简介

本项目是一个 **像素级不确定性感知视频超分** 的完整实现原型。

在前面 motion-based ROI 视频超分的基础上，进一步升级为：

### 三层递进架构

1. **第一层**：`full_heavy` - 全帧 BasicVSR++（基准）
2. **第二层**：`roi_heavy` - Motion-based ROI + bicubic 融合（轻量化）
3. **第三层**：`uncertainty_hybrid` - **像素级不确定性感知混合** ⭐ (新！)

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

### 2.1 三层递进方案

**第一层：full_heavy** - 全帧 BasicVSR++
- 基准方案，对每帧全面使用重模型推理
- 质量最好，但速度最慢

**第二层：roi_heavy** - Motion-based ROI + bicubic 融合
- 使用"运动强度"做 ROI，在不引入额外检测模型的前提下实现区域感知
- 相邻帧之间运动越强，越可能存在细节退化更明显的区域
- 用灰度差分、阈值和形态学操作得到轻量运动区域估计
- 非 ROI 区域用 bicubic 快速上采样，ROI 区域用 BasicVSR++ 重建
- 速度和质量的平衡

**第三层：uncertainty_hybrid** - 像素级不确定性感知混合 ⭐
- 在 roi_heavy 基础上进一步升级，采用 **像素级不确定性加权融合**
- 对每个像素计算不确定性值，根据不确定性决定使用保守重建还是纹理增强
- 完整融合公式：
  $$U(x,y) = \sigma\left(\frac{\alpha \cdot R + \beta \cdot T - \gamma \cdot S}{\tau}\right)$$
  其中 R 为时间残差（运动线索），T 为纹理复杂度（细节线索），S 为结构置信度（边缘保护线索）
- 最终像素融合：
  $$I_{fused} = (1-U) \cdot I_{basicvsr} + U \cdot I_{texture}$$
- 质量最好，同时保持较高速度

### 2.2 不确定性感知原理

不确定性映射基于三个线索的融合：

1. **时间残差 (R)**：相邻帧的运动残差，高残差表示不稳定区域
2. **纹理复杂度 (T)**：Laplacian 响应，高值表示细节丰富的区域
3. **结构置信度 (S)**：Sobel 梯度强度，高值表示边缘和结构明显的区域

融合的好处：
- 低纹理、低运动的平滑区域 → 低不确定性 → 信任 BasicVSR++ 保守重建
- 高纹理、高运动但低结构的细节区域 → 高不确定性 → 使用纹理增强分支
- 高结构的边缘区域 → 低不确定性 → 保护边缘质量

可视化输出：
- **热力图 (heatmap)**：Jet 色卡显示不确定性分布（蓝=低，红=高）
- **二值掩码 (binary_mask)**：按阈值二值化，白色=高不确定性，黑色=低不确定性

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
- `src/uncertainty.py`：不确定性计算（时间残差、纹理复杂度、结构置信度），不确定性平滑和可视化
- `src/fusion.py`：ROI 拼回整帧、可选 feather blending 和像素级不确定性加权融合
- `src/benchmark.py`：统计耗时、FPS、GPU 显存峰值和不确定性统计
- `main.py`：命令行入口，支持 `full_heavy`、`roi_heavy` 和 `uncertainty_hybrid` 三种模式

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

## 6. 配置说明

### 6.1 路径配置

默认配置文件是 `configs/default.yaml`。重点检查这些字段：

- `input_video_dir`：输入视频目录
- `basicvsr_root`：BasicVSR++ 仓库路径
- `basicvsr_config`：BasicVSR++ 配置文件
- `basicvsr_checkpoint`：BasicVSR++ 权重路径
- `input_video_path`：单个视频路径（优先级最高）
- `input_frames_dir`：帧序列输入目录
- `temp_dir`：临时文件输出目录
- `output_dir`：最终输出目录
- `video_output_dir`：视频输出目录
- `summary_dir`：统计摘要目录

默认配置已经指向 `data/` 目录，因此只要把视频放进 `roi_vsr_project/data/`，就会自动批处理。

### 6.2 不确定性参数配置

在 `configs/default.yaml` 的 `uncertainty` 段配置：

```yaml
uncertainty:
  alpha: 1.0                           # 时间残差权重（↑ 加重运动线索）
  beta: 0.8                            # 纹理复杂度权重（↑ 加重细节线索）
  gamma: 0.6                           # 结构置信度权重（↑ 更保护边缘）
  sigmoid_temperature: 1.0             # Sigmoid 温度（↓ 更尖锐的分界）
  spatial_smooth_ksize: 9              # 空间平滑核大小（必须奇数）
  temporal_smooth_lambda: 0.7          # 时间平滑系数 ∈ [0, 1]
  threshold: 0.5                       # 二值掩码阈值 ∈ [0, 1]
  save_heatmap: true                   # 保存热力图
  save_overlay: true                   # 保存二值掩码
```

**参数调优建议**：

| 现象 | 调整方向 | 原因 |
|------|--------|------|
| 热力图全是蓝色（低不确定性） | ↓ alpha/beta，↑ gamma | 不确定性过低，导致过度相信保守重建 |
| 热力图全是黄色/绿色/红色 | ↑ alpha/beta，↓ gamma | 不确定性过高，导致过度相信纹理增强 |
| 边缘模糊 | ↑ gamma | 增加边缘保护 |
| 细节丢失 | ↑ beta | 增加纹理复杂度权重 |
| 运动区域重建差 | ↑ alpha | 增加时间残差权重 |
| 不确定性分界不清晰 | ↓ sigmoid_temperature | 使 sigmoid 更陡峭 |

### 6.3 纹理增强分支配置

在 `texture_branch` 段配置纹理增强方法：

```yaml
texture_branch:
  method: "realesrgan"                 # 增强方法：realesrgan / unsharp / local_contrast / hybrid
  realesrgan:
    model_name: "RealESRGAN_x4plus"
    model_path: null                   # null 表示自动下载
    scale: 4                           # 必须与 VSR 上采样倍数一致
    tile: 0                            # 0 表示全帧推理
    tile_pad: 10                       # 分块推理的 padding
```

### 6.4 其他关键参数

```yaml
upscale_factor: 4                      # 上采样倍数
motion_threshold: 20.0                 # ROI 运动阈值
max_seq_len: 8                         # 最大序列长度（防止 GPU 显存溢出）
device: 0                              # GPU 设备编号
fps: 25.0                              # 输出视频帧率
save_intermediate: true                # 保存中间结果（便于调试）
save_videos: true                      # 生成输出视频
```

## 7. 三种运行模式

### 7.1 全帧重模型推理 (full_heavy)

```bash
python main.py --config configs/default.yaml --mode full_heavy
```

对输入视频的每一帧都使用 BasicVSR++ 进行全面重建，质量最好但速度最慢。

### 7.2 运动感知轻量化推理 (roi_heavy)

```bash
python main.py --config configs/default.yaml --mode roi_heavy
```

工作流程：
1. 读取原始低分辨率帧
2. 计算运动 ROI bbox
3. 全图先 bicubic 上采样
4. ROI 裁剪后送入 BasicVSR++
5. 将 ROI 增强结果粘回整帧

输出会按视频名分别写入 `results/output/<video_name>/roi_heavy/`。

### 7.3 像素级不确定性感知混合推理 (uncertainty_hybrid) ⭐ 新！

```bash
python main.py --config configs/default.yaml --mode uncertainty_hybrid
```

完整工作流程：
1. 读取输入帧序列
2. BasicVSR++ 保守重建（每帧）
3. 纹理增强分支处理（如 RealESRGAN）
4. **计算像素级不确定性映射**
   - 时间残差线索
   - 纹理复杂度线索
   - 结构置信度线索
5. **不确定性加权融合**
   - 每个像素根据不确定性值在两个分支间加权
   - U ≈ 0 → 信任 BasicVSR++ 保守重建
   - U ≈ 1 → 使用纹理增强分支
6. 时空平滑处理
7. 生成热力图和二值掩码可视化
8. 输出融合结果

输出会按视频名分别写入 `results/output/<video_name>/uncertainty_hybrid/`。

### 7.4 一键批处理（推荐）

```bash
python main.py --config configs/default.yaml
```

默认会对 `data/` 下的每个视频同时跑 `full_heavy`、`roi_heavy` 和 `uncertainty_hybrid`，并自动输出视频、帧和摘要结果。

## 9. 输出结果说明

### 9.1 各模式输出目录

运行完成后，默认会得到：

- `results/output/<video_name>/full_heavy/`：每个视频的 full_heavy 输出帧
- `results/output/<video_name>/roi_heavy/`：每个视频的 ROI-heavy 输出帧
- `results/output/<video_name>/uncertainty_hybrid/`：每个视频的 uncertainty_hybrid 输出帧
- `results/output/<video_name>/original_frames/`：原始输入帧的保存副本（若开启 `save_intermediate`）
- `results/video/<video_name>/`：自动导出的 mp4 视频，包括原始输入、full_heavy、roi_heavy、uncertainty_hybrid

### 9.2 不确定性可视化输出（仅 uncertainty_hybrid）

在 `results/tmp/uncertainty_hybrid/uncertainty_vis/` 目录下：

- **热力图 (heatmaps/)**
  - `heatmap_{i:08d}.png`：不确定性热力图，使用 Jet 色卡
    - 红色：高不确定性（推荐使用纹理增强）
    - 黄色/绿色：中等不确定性
    - 蓝色：低不确定性（推荐使用保守重建）

- **二值掩码 (overlays/)**
  - `binary_mask_{i:08d}.png`：不确定性阈值二值化
    - 白色像素：不确定性 > 阈值（默认 0.5）
    - 黑色像素：不确定性 ≤ 阈值

### 9.3 统计信息

- `results/output/<video_name>/summary/metrics.json`：每个视频的性能指标
  - `total_seconds`：总耗时
  - `avg_frame_seconds`：平均每帧耗时
  - `fps`：帧率
  - `peak_gpu_memory_mib`：GPU 显存峰值
  - `average_roi_area_ratio`：ROI 面积占比（仅 roi_heavy）
  - `avg_uncertainty`：平均不确定性（仅 uncertainty_hybrid）
  - `high_uncertainty_ratio`：高不确定性像素占比（仅 uncertainty_hybrid）
  - `psnr` / `ssim`：相对一致性指标
- `results/summary/run_summary_*.json`：本次批处理的总汇总

如果开启了 `save_intermediate: true`，中间结果会更完整，便于调试 ROI 是否稳定、裁剪是否正确、融合是否对齐。

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

### 11.1 不确定性计算方面

- 时间残差基于简单帧差，未考虑光流
- 纹理复杂度仅用 Laplacian 估计，可能对某些纹理过度敏感
- 结构置信度基于 Sobel 梯度，无法区分物体边界和纹理
- 参数 alpha/beta/gamma 需要手工调优，未采用自适应方案

### 11.2 融合方面

- ROI 估计只用帧差分，没有引入目标检测或分割模型
- 当前主流程优先实现 bbox 方案，mask 级融合还只是预留接口
- ROI 采用全局稳定框，适合原型验证，不适合复杂多目标场景
- 不确定性平滑使用简单高斯模糊和指数移动平均，未采用高级时空滤波

### 11.3 性能评估方面

- GPU 显存统计是 best-effort，跨子进程只能粗略采样
- PSNR / SSIM 不是对 GT 的质量评估，而是不同方案间的相对比较
- 无绝对质量基准（GT），仅有相对一致性指标

## 12. 后续扩展方向

### 推荐的改进方向

1. **不确定性计算增强**
   - 引入光流估计替代帧差分
   - 采用多尺度纹理分析
   - 加入帧间一致性线索
   - 实现自适应参数学习

2. **融合策略优化**
   - 支持多 ROI 和分块推理
   - 实现 mask 级别融合与 feather blending
   - 添加时序 ROI 跟踪和框平滑
   - 自适应 ROI 扩展机制

3. **模型替换**
   - 将 `bicubic` 背景替换为轻量版 BasicVSR++ 或其他轻模型
   - 支持不同纹理增强方法的动态选择

4. **质量评估**
   - 如果后续有 GT，切换 PSNR / SSIM 参考到真实高质量标注
   - 添加感知质量指标（LPIPS、FID 等）

## 13. 快速开始

### 13.1 最小化设置

1. 准备输入视频（放入 `data/` 目录）
2. 确保 BasicVSR++ 环境已配置
3. 一键运行：

```bash
python main.py --config configs/default.yaml
```

这会自动生成三种模式的结果（full_heavy、roi_heavy、uncertainty_hybrid）。

### 13.2 只运行不确定性混合模式

```bash
python main.py --config configs/default.yaml --mode uncertainty_hybrid
```

### 13.3 查看不确定性热力图

运行完成后查看：
```
results/tmp/uncertainty_hybrid/uncertainty_vis/heatmaps/
```

蓝色区域表示低不确定性（信任 BasicVSR++），红色区域表示高不确定性（使用纹理增强）。

### 13.4 调试和优化

如果输出质量不理想：

1. **检查热力图**：看不确定性分布是否合理
2. **调整参数**：根据参数调优表修改 alpha/beta/gamma
3. **查看中间结果**：启用 `save_intermediate: true` 观察 ROI 和各分支结果
4. **检查日志**：查看每个模块的运行时间和 GPU 显存使用情况
