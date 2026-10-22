# SPS OCTMNIST SNN

本项目提供 OCTMNIST 脉冲神经网络（SNN）的代码和数据索引，并以五个随机种子比较浮点权重、理想量化权重和实测电导映射权重。

## 数据与划分

使用 MedMNIST v2、数据发布版本 v3.0 的 28 × 28 `octmnist.npz`（[Zenodo 10519652](https://zenodo.org/records/10519652)，CC BY 4.0）。预期 MD5 为 `c68d92d5b585d8d81f7112f81e2d0842`。将文件保存为 `data/octmnist.npz`。

```bash
mkdir -p data
curl -L "https://zenodo.org/records/10519652/files/octmnist.npz?download=1" -o data/octmnist.npz
echo "c68d92d5b585d8d81f7112f81e2d0842  data/octmnist.npz" | md5sum -c -
```

在 Windows 中可用 `(Get-FileHash .\data\octmnist.npz -Algorithm MD5).Hash` 校验。加载程序还会检查划分大小、图像尺寸和标签范围。

| 划分 | 纳入数／官方总数 | CNV | DME | Drusen | Normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练集 | 97,477／97,477 | 33,484 | 10,213 | 7,754 | 46,026 |
| 验证集 | 9,483／10,832 | 2,759 | 1,054 | 732 | 4,938 |
| 测试集 | 1,000／1,000 | 250 | 250 | 250 | 250 |

`data/split_indices.csv` 记录每张图像在官方 NPZ 划分内从零开始的行索引、标签、纳入标志和原始像素 SHA-256。`data/validation_excluded_indices.csv` 记录与训练图像的原始 28 × 28 数组完全相同、因而被排除的 1,349 条验证集索引。训练集和测试集保留官方索引及顺序；各划分内部的重复图像未去重。筛选后的跨划分原始像素哈希审计未发现完全相同的图像。归档中没有患者标识，无法据此判断患者级重叠。运行 `python export_validation_exclusions.py --dataset data/octmnist.npz --overwrite` 可重新生成派生索引，不修改 NPZ。

不进行类别平衡子采样、过采样或有放回采样；每轮使用全部纳入的训练图像各一次。类别不平衡仅通过损失权重处理：`w_c=(1/sqrt(n_c))/mean_k(1/sqrt(n_k))`。CNV、DME、Drusen、Normal 的权重依次为 `0.69666175, 1.26143235, 1.44769719, 0.59420871`。测试集不参与模型选择或调参。

## 输入与模型

原始 uint8 图像除以 255。输入包含原图和增强图两个通道；增强图先计算 `z=clip((x-mean(x))/(2.45*max(std(x),0.08))+0.5,0,1)`，再计算 `clip(z+0.28*(z-smooth3x3(z)),0,1)`，其中 `smooth3x3` 为 3 × 3 均值滤波。仅训练时增强：水平翻转（`p=0.5`）；整批共享的每轴 −1、0 或 +1 像素平移（`p=0.6`，零填充）；逐图对比度 `[0.92,1.08]`；亮度 `[−0.02,0.02]`；高斯噪声（标准差 `0.006`，`p=0.25`）。验证和测试时不增强，也不进行翻转平均。

对归一化强度 `x`，相位编码采用 `p=clip(1-exp(-r_in*dt*x),0,0.97)`，其中 `r_in=9.0e5 Hz`、`dt=1 µs`、初始相位为 `0.5`。相位达到 1 时发放脉冲，并减去 1。模型训练与评估均使用 16 个时间步（16 µs）。

LIF 更新为 `u_t=βu_(t−1)+I_t`、`s_t=H(u_t−θ)`，随后 `u_t←u_t−s_tθ`。实测 `τ=8.7 µs` 对应 `β=exp(−dt/τ)=exp(−1/8.7)=0.891417373`。复位为减阈值式软复位，不将膜电位钳位至零。八个 LIF 阈值依次为 `0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.05`。

网络结构：双通道输入 → Conv48-LIF × 2 → 平均池化 → Conv96-LIF × 2 → 平均池化 → Conv192-LIF × 3 → 平均池化 → FC384-LIF → FC4。卷积采用 3 × 3 核、填充 1 和批归一化；FC384 后采用层归一化。同宽卷积层的残差脉冲电流系数为 `0.20, 0.20, 0.15, 0.15`。四个输出神经元积累带泄漏的时序证据，不设置输出发放阈值。

## 训练与权重映射

| 设置 | 数值 |
| --- | --- |
| 优化器 | AdamW；余弦学习率调度 |
| 学习率 | 共享基线 `8.0e-4`；各后续分支 `2.0e-4` |
| 权重衰减 | 基线 `1.5e-4`；后续分支非电导参数减半；差分电导为零 |
| 批量大小 | 全局训练 96；评估 384 |
| 训练轮数 | 共享连续权重训练 70 轮，各分支继续 30 轮（每组有效训练 100 轮） |
| 正则化 | 标签平滑 `0.035`；梯度裁剪 `3.0`；发放率超出 `[0.035,0.36]` 时脉冲惩罚 `2.5e-4` |
| 随机种子 | `19, 29, 39, 49, 59` |

同一随机种子的三个分支均从第 70 轮最终检查点 `base_epoch_70.pt` 出发，分别使用新的 AdamW 优化器和相同的后续学习率调度。`float` 继续训练连续权重；`ideal` 和 `measured` 使用直通估计器进行量化感知训练（QAT）。第 71–100 轮中，按 `0.66×验证准确率 + 0.19×验证宏平均 F1 + 0.15×验证集最低类别召回率` 选取最佳检查点；选定后才计算测试指标。

`line_LTP_LTD.csv` 含有 80 个实测 LTP 状态和 80 个实测 LTD 状态，单位为 S，范围为 `3.318e-5`–`4.19208e-4 S`。首行数据（`Number=1`）是零脉冲状态（`k=0`），末行为 `k=79`。有符号突触使用 `ΔG=G+−G−`、`D=G_max−G_min` 和层权重 `w=sqrt(6/fan_in)*ΔG/D`；可训练影子参数 `ΔG` 投影至 `[−D,D]`。

理想映射采用 80 个均匀间隔、完全可逆的状态 `G_k=G_min+kD/79`，形成 159 个有符号差分等级。实测映射保留 LTP/LTD 分支及脉冲索引，从 `160²=25,600` 种有序实测状态配对中选择最接近目标电导差的一对。量化前向计算采用选中的差值，反向计算采用直通估计器；映射后推理使用选中的实际电导。偏置、归一化参数、阈值等仍为数字模型参数。

## 运行与输出

安装适合目标 GPU 的 CUDA 版 PyTorch；依赖见 `requirements.txt`。以下命令运行五个随机种子和三个分支：

```bash
python run_ablation.py --dataset data/octmnist.npz --output-dir results/ablation
```

运行前可用 `python run_ablation.py --dataset data/octmnist.npz --dry-run` 检查输入与命令。直接运行 `main.py` 仅训练一个 70 轮基线；后续分支须通过 `--base-checkpoint` 指定同种子的 `base_epoch_70.pt`。双 GPU 可增加 `--nproc-per-node 2`，训练批量 96 仍为全局大小。仅在设置匹配时使用 `--reuse-completed`，否则使用新的输出目录。

`results/ablation/` 保存各随机种子的基线与 `float`、`ideal`、`measured` 分支，包括 `run_manifest.json`、`run.log`、训练历史、发放率表、划分指标、混淆矩阵和检查点。`best.pt` 依据验证集选择；实测分支另有记录所选 `G+/G−` 状态的 `programmed.pt`。汇总表为 `ablation_per_seed.csv` 和 `ablation_mean_std.csv`。图 5c–g 的源数据与图片可用以下命令重建：

```bash
python reproduce_fig5_revised.py --dataset data/octmnist.npz --results-dir results/ablation --output-dir results/figure5_revised
```

图件输出目录应为空。发布文件清单见 `RELEASE_CHECKLIST.md`。

## 已完成的五种子测试结果

官方 1,000 张图像测试集上、按验证集选定检查点后的结果；均值 ± 样本标准差（`ddof=1`）：

| 权重条件 | 准确率 | 宏平均 F1 | Drusen 召回率 |
| --- | ---: | ---: | ---: |
| 浮点权重 | 76.40 ± 0.35% | 73.43 ± 0.90% | 29.76 ± 4.17% |
| 理想量化 | 76.54 ± 1.38% | 73.61 ± 1.98% | 30.32 ± 4.74% |
| 实测电导映射 | 76.86 ± 0.77% | 74.09 ± 1.34% | 31.76 ± 4.36% |

训练环境：Python `3.12.12`、NumPy `2.3.5`、PyTorch `2.8.0+cu128`（CUDA 运行时 `12.8`）、NVIDIA GeForce RTX 4090。
