# 多攻击评估实验指南

本文档介绍如何使用 `multi_attack_eval` 模式进行对抗攻击评估实验。

## 功能概述

`multi_attack_eval` 模式可以：
- 运行多种对抗攻击（共15种）
- 比较攻击后准确率 vs FFT Top-K 恢复后准确率
- 按调制类型和信噪比(SNR)分类统计
- 生成频域对比图和IQ分布图

## 支持的攻击类型

| 攻击名称 | 类型 | 说明 |
|---------|------|------|
| fgsm | 单步攻击 | Fast Gradient Sign Method |
| pgd | 迭代攻击 | Projected Gradient Descent |
| bim | 迭代攻击 | Basic Iterative Method |
| cw | 优化攻击 | Carlini & Wagner L2攻击 |
| deepfool | 优化攻击 | 最小扰动攻击 |
| apgd | 自适应攻击 | Auto-PGD |
| mifgsm | 动量攻击 | Momentum Iterative FGSM |
| rfgsm | 随机攻击 | Random FGSM |
| upgd | 通用攻击 | Universal PGD |
| eotpgd | 期望攻击 | Expectation Over Transformation PGD |
| vmifgsm | 方差攻击 | Variance-tuning MI-FGSM |
| vnifgsm | 方差攻击 | Variance-tuning NI-FGSM |
| jitter | 抖动攻击 | Jitter攻击 |
| ffgsm | 快速攻击 | Fast FGSM |
| pgdl2 | L2攻击 | PGD L2范数攻击 |

## 基本命令格式

```bash
python main.py --mode multi_attack_eval --dataset <数据集> --ckpt_path <模型路径> \
    [--mod_filter <调制类型>] [--snr_filter <信噪比>] \
    [--attack_list <攻击列表>] [--plot_iq] [--plot_freq]
```

## 常用实验示例

### 1. 单一调制类型 + 单一SNR + 所有攻击 + IQ分布图

**场景**: 测试 QAM64 在 SNR=0dB 下所有攻击的效果，并生成IQ分布图

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --plot_iq
```

**输出**:
- CSV文件: `inference/<dataset>_*/result/multi_attack_snr_mod_eval.csv`
- IQ图: `inference/<dataset>_*/result/iq_plots/`

### 2. 单一调制类型 + 单一SNR + 指定攻击

**场景**: 只测试 FGSM 攻击

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --attack_list fgsm --plot_iq
```

### 3. 同时生成频域图和IQ图

**场景**: 同时查看频域变化和IQ分布变化

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --attack_list fgsm --plot_iq --plot_freq
```

**输出**:
- IQ图: `inference/<dataset>_*/result/iq_plots/`
- 频域图: `inference/<dataset>_*/result/freq_plots/`

### 4. 测试多种攻击

**场景**: 比较 FGSM、PGD、CW 三种攻击

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --attack_list "fgsm,pgd,cw" --plot_iq
```

### 5. 高SNR环境测试

**场景**: 在 SNR=18dB（高信噪比）下测试

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 18 --plot_iq
```

### 6. 测试所有调制类型（固定SNR）

**场景**: 在 SNR=0dB 下测试所有调制类型

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --snr_filter 0 --plot_iq
```

### 7. 测试所有SNR（固定调制类型）

**场景**: 测试 QAM64 在所有SNR下的表现

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --plot_iq
```

### 8. 完整测试（所有调制 × 所有SNR × 所有攻击）

**警告**: 这会花费很长时间！

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --plot_iq --plot_freq
```

### 9. 快速测试（限制样本数）

**场景**: 快速验证，每个(SNR, 调制)组合只用50个样本

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --eval_limit_per_cell 50 --plot_iq
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | - | 数据集: 2016.10a, 2016.10b, 2018.01a |
| `--ckpt_path` | ./checkpoint | 模型检查点路径 |
| `--mod_filter` | None | 过滤调制类型，如 QAM64, QPSK, BPSK |
| `--snr_filter` | None | 过滤SNR值，如 0, 10, 18 |
| `--attack_list` | 所有15种 | 逗号分隔的攻击列表 |
| `--attack_eps` | 0.3 | Linf攻击的epsilon值 |
| `--plot_iq` | False | 生成IQ分布对比图 |
| `--plot_freq` | False | 生成频域对比图 |
| `--plot_n_samples` | 3 | 单独绘制的样本数量 |
| `--eval_limit_per_cell` | None | 每个(SNR,调制)组合的最大样本数 |

## CW攻击参数调整

CW攻击可以通过以下参数调整强度：

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --attack_list cw \
    --cw_c 10.0 --cw_steps 200 --cw_lr 0.005 --plot_iq
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cw_c` | 10.0 | 置信度参数（越高攻击越强） |
| `--cw_steps` | 200 | 优化步数（越多攻击越强但越慢） |
| `--cw_lr` | 0.005 | 优化学习率 |

## 输出文件说明

### CSV结果文件

路径: `inference/<dataset>_*/result/multi_attack_snr_mod_eval.csv`

| 列名 | 说明 |
|------|------|
| attack | 攻击名称 |
| snr | 信噪比 |
| modulation | 调制类型 |
| n_samples | 样本数量 |
| attack_acc | 攻击后准确率（恢复前） |
| top10_acc | FFT Top-10 恢复后准确率 |
| top20_acc | FFT Top-20 恢复后准确率 |

### IQ分布图

路径: `inference/<dataset>_*/result/iq_plots/`

| 文件名 | 说明 |
|--------|------|
| `<attack>_<mod>_snr<snr>_iq_sample1.png` | 单个样本的IQ散点图 |
| `<attack>_<mod>_snr<snr>_iq_all.png` | 所有样本聚合的IQ散点图 |
| `<attack>_<mod>_snr<snr>_iq_density.png` | IQ密度直方图（显示分布差异） |

### 频域对比图

路径: `inference/<dataset>_*/result/freq_plots/`

| 文件名 | 说明 |
|--------|------|
| `<attack>_<mod>_snr<snr>_sample1.png` | 单个样本的频谱图 |
| `<attack>_<mod>_snr<snr>_avg.png` | 平均频谱图 |
| `<attack>_<mod>_snr<snr>_overlay.png` | 干净 vs 对抗频谱叠加图 |

## 可用的调制类型

### 2016.10a 数据集（11类）
QAM16, QAM64, 8PSK, WBFM, BPSK, CPFSK, AM-DSB, GFSK, PAM4, QPSK, AM-SSB

### 2016.10b 数据集（10类）
QAM16, QAM64, 8PSK, WBFM, BPSK, CPFSK, AM-DSB, GFSK, PAM4, QPSK

### 2018.01a 数据集（24类）
OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, OQPSK

## 可用的SNR值

### 2016.10a / 2016.10b
-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 (dB)

## 常见问题

### Q: 为什么IQ分布变化很大但准确率没有下降？

A: AWN模型使用小波分解提取频域特征。FGSM等攻击在时域添加扰动，可能不会影响模型依赖的小波系数。建议：
1. 增大epsilon: `--attack_eps 0.5`
2. 使用CW攻击（针对决策边界优化）
3. 查看频域图确认频谱变化

### Q: 如何加速实验？

A: 使用 `--eval_limit_per_cell 50` 限制每组样本数量。

### Q: 如何只运行特定几种攻击？

A: 使用 `--attack_list "fgsm,pgd,cw"` 指定攻击列表。
