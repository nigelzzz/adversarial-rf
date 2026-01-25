# 多攻击评估实验指南

本文档介绍如何使用 `multi_attack_eval` 和 `sigguard_eval` 模式进行对抗攻击评估实验。

## 功能概述

### multi_attack_eval 模式
- 运行多种对抗攻击（共17种）
- 比较攻击后准确率 vs FFT Top-K 恢复后准确率
- 按调制类型和信噪比(SNR)分类统计
- 生成频域对比图和IQ分布图

### sigguard_eval 模式（新增）
- 生成类似学术论文的表格输出
- 比较"Disabled"（无防御）和"Enabled"（FFT Top-K防御）准确率
- 自动生成IQ分布对比图
- 适合论文结果展示

## 支持的攻击类型（17种）

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
| eadl1 | 弹性网攻击 | EAD L1正则化攻击 |
| eaden | 弹性网攻击 | EAD Elastic Net攻击 |

## Epsilon配置（重要）

RF IQ信号与图像不同，需要不同的epsilon值：

### 问题背景
- IQ信号在 [-1, 1] 范围内，但实际幅度约 ±0.02
- 转换到 [0, 1] 后，信号只占约2%的范围
- 图像常用的 eps=0.3 对IQ数据来说太大（是信号幅度的15倍）

### 归一化模式 (`--ta_box`)

| 模式 | 映射方式 | Epsilon含义 | 推荐用途 |
|------|---------|------------|---------|
| `unit` | `(x+1)/2` | [0,1]空间中的绝对值 | 简单，需要小eps (~0.03) |
| `minmax` | 每样本min-max到[0,1] | 相对于信号范围 | 更直观的eps值 |

### 推荐Epsilon值

| 模式 | Epsilon | 效果 |
|------|---------|------|
| `unit` | 0.01-0.03 | 轻微扰动 |
| `unit` | 0.05-0.1 | 中等攻击 |
| `minmax` | 0.05-0.1 | 轻微扰动 |
| `minmax` | 0.2-0.3 | 中等攻击 |

---

## SigGuard评估模式（推荐用于论文）

### 基本用法

```bash
# 运行所有17种攻击，生成表格和IQ图
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint

# 使用minmax模式获得更好的攻击效果
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --ta_box minmax --attack_eps 0.1

# 指定攻击列表
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --attack_list "cw,fgsm,pgd,eadl1,eaden"

# 调整防御Top-K值
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --sigguard_topk 10

# 快速测试（限制样本数）
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --eval_limit 1000

# 不生成IQ图（更快）
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --no_plot_iq
```

### 输出格式

```
  AWN - SigGuard Evaluation (Top-50)
  ==================================================
  Sample Type         Disabled      Enabled
  --------------------------------------------------
  Intact              92.61%        92.20%
  FGSM                 7.20%         9.32%
  PGD                  5.10%        12.45%
  CW                   0.86%        80.43%
  EADL1                0.00%        78.34%
  EADEN                0.00%        74.01%
  ...                   ...           ...
  ==================================================
```

### SigGuard参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sigguard_topk` | 50 | FFT Top-K防御的K值 |
| `--no_plot_iq` | False | 禁用IQ分布图生成 |
| `--eval_limit` | None | 限制测试样本数量 |

### 输出文件

- CSV: `inference/<dataset>_*/result/sigguard_eval.csv`
- 表格: `inference/<dataset>_*/result/sigguard_eval_table.txt`
- IQ图: `inference/<dataset>_*/result/iq_plots/`
  - `<attack>_iq_sample1.png` - 单样本散点图
  - `<attack>_iq_all.png` - 聚合散点图
  - `<attack>_iq_density.png` - 密度直方图

---

## Multi-Attack评估模式

### 基本命令格式

```bash
python main.py --mode multi_attack_eval --dataset <数据集> --ckpt_path <模型路径> \
    [--mod_filter <调制类型>] [--snr_filter <信噪比>] \
    [--attack_list <攻击列表>] [--plot_iq] [--plot_freq]
```

### 常用实验示例

#### 1. 单一调制类型 + 单一SNR + 所有攻击 + IQ分布图

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --plot_iq
```

#### 2. 使用minmax模式获得更好的攻击效果

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --ta_box minmax --attack_eps 0.1 --plot_iq
```

#### 3. 同时生成频域图和IQ图

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --attack_list fgsm --plot_iq --plot_freq
```

#### 4. 测试EAD攻击

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --attack_list "eadl1,eaden" --plot_iq
```

#### 5. 快速测试（限制样本数）

```bash
python main.py --mode multi_attack_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --mod_filter QAM64 --snr_filter 0 --eval_limit_per_cell 50 --plot_iq
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | - | 数据集: 2016.10a, 2016.10b, 2018.01a |
| `--ckpt_path` | ./checkpoint | 模型检查点路径 |
| `--mod_filter` | None | 过滤调制类型，如 QAM64, QPSK, BPSK |
| `--snr_filter` | None | 过滤SNR值，如 0, 10, 18 |
| `--attack_list` | 所有17种 | 逗号分隔的攻击列表 |
| `--attack_eps` | 0.03 | Linf攻击的epsilon值 |
| `--ta_box` | unit | 归一化模式: unit 或 minmax |
| `--plot_iq` | False | 生成IQ分布对比图 |
| `--plot_freq` | False | 生成频域对比图 |
| `--plot_n_samples` | 3 | 单独绘制的样本数量 |
| `--eval_limit_per_cell` | None | 每个(SNR,调制)组合的最大样本数 |

---

## 攻击参数调整

### CW攻击参数

```bash
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --attack_list cw --cw_c 10.0 --cw_steps 200 --cw_lr 0.005
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cw_c` | 10.0 | 置信度参数（越高攻击越强） |
| `--cw_steps` | 200 | 优化步数（越多攻击越强但越慢） |
| `--cw_lr` | 0.005 | 优化学习率 |

### EAD攻击参数（EADL1, EADEN）

```bash
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
    --attack_list "eadl1,eaden" --ead_kappa 5 --ead_max_iterations 100
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ead_kappa` | 0 | 置信度参数（越高攻击越强） |
| `--ead_lr` | 0.01 | 学习率 |
| `--ead_max_iterations` | 100 | 最大迭代次数 |
| `--ead_binary_search_steps` | 9 | 二分搜索步数 |
| `--ead_initial_const` | 0.001 | 初始常数 |
| `--ead_beta` | 0.001 | L1/L2权衡参数 |

---

## 输出文件说明

### CSV结果文件 (multi_attack_eval)

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

| 文件名 | 说明 |
|--------|------|
| `<attack>_iq_sample1.png` | 单个样本的IQ散点图 |
| `<attack>_iq_all.png` | 所有样本聚合的IQ散点图 |
| `<attack>_iq_density.png` | IQ密度直方图（显示分布差异） |

### 频域对比图

| 文件名 | 说明 |
|--------|------|
| `<attack>_<mod>_snr<snr>_sample1.png` | 单个样本的频谱图 |
| `<attack>_<mod>_snr<snr>_avg.png` | 平均频谱图 |
| `<attack>_<mod>_snr<snr>_overlay.png` | 干净 vs 对抗频谱叠加图 |

---

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

---

## 常见问题

### Q: 为什么攻击后准确率没有明显下降？

A: 可能是epsilon设置问题。RF IQ数据的幅度很小（约±0.02），建议：
1. 使用minmax模式: `--ta_box minmax --attack_eps 0.1`
2. 或使用小的绝对epsilon: `--attack_eps 0.03`
3. 验证方法：有效攻击应使准确率从~90%降至20-40%

### Q: 如何加速实验？

A:
- 限制样本数: `--eval_limit 1000` 或 `--eval_limit_per_cell 50`
- 禁用IQ图: `--no_plot_iq`
- 选择部分攻击: `--attack_list "fgsm,pgd,cw"`

### Q: sigguard_eval 和 multi_attack_eval 有什么区别？

A:
- `sigguard_eval`: 输出简洁的表格（Disabled/Enabled），适合论文展示
- `multi_attack_eval`: 按SNR和调制类型细分，输出详细CSV，适合深入分析

### Q: APGD攻击报错怎么办？

A: APGD对批量大小敏感。代码已自动处理，如果批量攻击失败会回退到单样本处理（较慢但可靠）。

### Q: 如何选择防御的Top-K值？

A:
- Top-10: 较强防御，可能影响正常准确率
- Top-20: 中等防御，推荐
- Top-50: 较弱防御，对正常准确率影响小

```bash
# sigguard_eval 使用 --sigguard_topk
python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint \
  --sigguard_topk 20

# multi_attack_eval 固定使用 Top-10 和 Top-20 两列输出
```



```bash 
$ python main.py --mode sigguard_eval --dataset 2016.10a --ckpt_path ./checkpoint --mod_filter QAM64 --snr_filter 0  --ta_box minmax --sigguard_topk 10 --plot_freq --plot_iq
====================
dataset : 2016.10a
base_dir : inference
epochs : 100
batch_size : 128
patience : 10
milestone_step : 3
gamma : 0.5
lr : 0.001
num_classes : 11
num_level : 1
regu_details : 0.01
regu_approx : 0.01
kernel_size : 3
in_channels : 64
latent_dim : 320
monitor : acc
test_batch_size : 64
classes : {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4, b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
cfg_dir : inference/2016.10a_121
model_dir : inference/2016.10a_121/models
log_dir : inference/2016.10a_121/log
result_dir : inference/2016.10a_121/result
mode : sigguard_eval
seed : 2022
device : cuda
ckpt_path : ./checkpoint
num_workers : 0
Draw_Confmat : True
Draw_Acc_Curve : True
mod_filter : QAM64
snr_filter : 0
attack : cw
cw_c : 1.0
cw_kappa : 0.0
cw_steps : 100
cw_lr : 0.01
cw_targeted : False
cw_scale : None
lowpass : True
lowpass_kernel : 17
spec_type : cw_tone
spec_eps : 0.1
spec_jnr_db : None
tone_freq : None
spec_band_low : 0.05
spec_band_high : 0.25
spec_mask_path : None
eval_limit : None
attack_backend : torchattacks
ta_box : minmax
spec_mask_out : None
def_band_low : None
def_band_high : None
def_mask_path : None
cmp_defense : False
def_notch_depth : 1.0
def_notch_trans : 3
def_hp_order : 1
def_auto_fmax : 0.08
def_auto_ref_low : 0.15
def_auto_ref_high : 0.5
def_auto_tau : 2.0
def_auto_max_width : 3
def_auto_depth_max : 0.8
def_auto_trans : 4
defense : none
def_ens_depths : 0.55,0.6,0.65
def_ens_trans : 4
def_topk : 50
def_topk_percent : None
detector_ckpt : None
detector_threshold : 0.004468164592981338
detector_norm_offset : 0.02
detector_norm_scale : 0.04
det_epochs : 10
det_batch_size : 256
det_lr : 0.001
det_wd : 0.0
det_patience : 5
det_train_limit : 20000
det_calib_quantile : 0.9
snr_min : None
freq_percents : [0.1, 0.2, 0.3, 0.4, 0.5]
dir_name : None
attack_list : None
eval_limit_per_cell : None
attack_eps : 0.03
plot_freq : True
plot_iq : True
plot_n_samples : 3
sigguard_topk : 10
no_plot_iq : False
ead_kappa : 0
ead_lr : 0.01
ead_max_iterations : 100
ead_binary_search_steps : 9
ead_initial_const : 0.001
ead_beta : 0.001
====================
>>> total params: 0.12M
********************
Signals.shape: [1000, 2, 128]
Labels.shape: [1000]
********************
Signal_train.shape: [600, 2, 128]
Signal_val.shape: [200, 2, 128]
Signal_test.shape: [200, 2, 128]
********************
Running SigGuard evaluation on 200 samples
Attacks: ['fgsm', 'pgd', 'bim', 'cw', 'deepfool', 'apgd', 'mifgsm', 'rfgsm', 'upgd', 'eotpgd', 'vmifgsm', 'vnifgsm', 'jitter', 'ffgsm', 'pgdl2', 'eadl1', 'eaden']
FFT Top-K defense with K=10
Using ta_box=minmax normalization, eps=0.03
IQ plots will be saved to: inference/2016.10a_121/result/iq_plots

=== Intact (Clean) ===
Intact: Disabled=93.00%, Enabled=97.00%

=== FGSM ===
FGSM: Disabled=6.00%, Enabled=95.00%
Generating IQ plots for fgsm...

=== PGD ===
PGD: Disabled=0.00%, Enabled=93.50%
Generating IQ plots for pgd...

=== BIM ===
BIM: Disabled=0.00%, Enabled=94.00%
Generating IQ plots for bim...

=== CW ===
CW: Disabled=0.00%, Enabled=96.00%
Generating IQ plots for cw...

=== DEEPFOOL ===
DEEPFOOL: Disabled=85.00%, Enabled=95.50%
Generating IQ plots for deepfool...

=== MIFGSM ===
MIFGSM: Disabled=0.00%, Enabled=94.00%
Generating IQ plots for mifgsm...

=== RFGSM ===
RFGSM: Disabled=0.00%, Enabled=93.00%
Generating IQ plots for rfgsm...

=== UPGD ===
UPGD: Disabled=0.00%, Enabled=94.00%
Generating IQ plots for upgd...

=== EOTPGD ===
EOTPGD: Disabled=0.00%, Enabled=93.50%
Generating IQ plots for eotpgd...

=== VMIFGSM ===
VMIFGSM: Disabled=0.00%, Enabled=93.00%
Generating IQ plots for vmifgsm...

=== VNIFGSM ===
VNIFGSM: Disabled=0.00%, Enabled=92.50%
Generating IQ plots for vnifgsm...

=== APGD ===
APGD: Disabled=0.00%, Enabled=93.00%
Generating IQ plots for apgd...

=== JITTER ===
JITTER: Disabled=1.50%, Enabled=92.50%
Generating IQ plots for jitter...

=== FFGSM ===
FFGSM: Disabled=9.00%, Enabled=94.50%
Generating IQ plots for ffgsm...

=== PGDL2 ===
PGDL2: Disabled=0.50%, Enabled=93.00%
Generating IQ plots for pgdl2...

=== EADL1 ===
EADL1: Disabled=0.00%, Enabled=96.00%
Generating IQ plots for eadl1...

=== EADEN ===
EADEN: Disabled=0.00%, Enabled=96.50%
Generating IQ plots for eaden...

  AWN - SigGuard Evaluation (Top-10)
  ==================================================
  Sample Type         Disabled      Enabled
  --------------------------------------------------
  Intact                93.00%       97.00%
  FGSM                   6.00%       95.00%
  PGD                    0.00%       93.50%
  BIM                    0.00%       94.00%
  CW                     0.00%       96.00%
  DEEPFOOL              85.00%       95.50%
  MIFGSM                 0.00%       94.00%
  RFGSM                  0.00%       93.00%
  UPGD                   0.00%       94.00%
  EOTPGD                 0.00%       93.50%
  VMIFGSM                0.00%       93.00%
  VNIFGSM                0.00%       92.50%
  JITTER                 1.50%       92.50%
  FFGSM                  9.00%       94.50%
  PGDL2                  0.50%       93.00%
  EADL1                  0.00%       96.00%
  EADEN                  0.00%       96.50%
  ==================================================


  AWN - SigGuard Evaluation (Top-10)
  ==================================================
  Sample Type         Disabled      Enabled
  --------------------------------------------------
  Intact                93.00%       97.00%
  FGSM                   6.00%       95.00%
  PGD                    0.00%       93.50%
  BIM                    0.00%       94.00%
  CW                     0.00%       96.00%
  DEEPFOOL              85.00%       95.50%
  MIFGSM                 0.00%       94.00%
  RFGSM                  0.00%       93.00%
  UPGD                   0.00%       94.00%
  EOTPGD                 0.00%       93.50%
  VMIFGSM                0.00%       93.00%
  VNIFGSM                0.00%       92.50%
  JITTER                 1.50%       92.50%
  FFGSM                  9.00%       94.50%
  PGDL2                  0.50%       93.00%
  EADL1                  0.00%       96.00%
  EADEN                  0.00%       96.50%
  ==================================================

Saved results to: inference/2016.10a_121/result/sigguard_eval.csv
Saved table to: inference/2016.10a_121/result/sigguard_eval_table.txt
IQ plots saved to: inference/2016.10a_121/result/iq_plots


```
- iq data distribution can reference <mod>_iq_all.png, e.g., bim_iq_all.png
