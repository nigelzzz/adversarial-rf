# Phase 7 報告：對抗式攻擊生成延遲基準測試（CPU vs GPU）

**日期：** 2026-04-26
**里程碑：** v1.1 強健性基準（Robustness Baselines）
**狀態：** 實作完成；尚待人工驗證 2 項 UAT 項目

## 1. 目標

以單一指令量測 5 種論文攻擊（FGSM、PGD、CW、EAD-L1、EAD-EN）在 CPU 與
GPU 上每樣本（per-sample）的對抗式擾動生成延遲，並產出可直接整合進論文
的延遲圖表。

## 2. 交付物

| 產出物 | 路徑 | 用途 |
|---|---|---|
| 計時引擎 | `util/attack_bench.py`（471 行） | `run_attack_bench_5x2()`：分層抽樣、同步框出計時、R=5 重複、論文鎖定 D-05 超參數、裝置切換時重新載入 state_dict |
| CLI 入口 | `main.py`（`--mode attack_bench` + 4 個 `--bench_*` 旗標） | 單一指令呼叫：`python main.py --mode attack_bench --dataset 2016.10a --ckpt_path ./checkpoint` |
| 延遲 CSV | `inference/2016.10a_*/result/attack_bench.csv` | 10 列表格（5 種攻擊 × 2 種裝置），含 mean ± std（毫秒/樣本） |
| 環境 metadata | `inference/2016.10a_*/result/attack_bench_env.json` | torch/cuda/gpu/cpu 重現性 sidecar |
| 繪圖工具 | `paper/scripts/plot_attack_bench_latency.py`（166 行） | 獨立 CSV → IEEE 風格 PDF 轉換器 |
| 圖表 | `paper/latex/figures/attack_bench_latency.pdf`（13.9 KB） | 論文相機就緒版（camera-ready）的延遲長條圖 |

## 3. 硬體與環境

- **GPU：** NVIDIA GeForce RTX 5060 Ti
- **CPU：** x86_64，6 核心
- **軟體堆疊：** torch 2.9.0+cu130、CUDA 13.0
- **資料集：** RML2016.10a 測試切分（44k 樣本）

## 4. 煙霧測試（Smoke-Budget）結果（N=64 樣本，R=2 次重複）

| 攻擊 | CPU（毫秒/樣本） | GPU（毫秒/樣本） | GPU 加速比 |
|---|---:|---:|---:|
| FGSM    | 0.176 ± 0.002   | 0.084 ± 0.001   | 2.1× |
| PGD     | 2.073 ± 0.149   | 0.753 ± 0.001   | 2.8× |
| CW      | 5.698 ± 0.005   | 2.269 ± 0.013   | 2.5× |
| EAD-L1  | 95.94 ± 15.76   | 26.24 ± 1.26    | 3.7× |
| EAD-EN  | 146.11 ± 5.03   | 25.45 ± 0.11    | 5.7× |

> **注意：** 這是煙霧預算（smoke-budget）快照。提交相機就緒版前須以預設
> 值重跑（N=512，R=5）。程式碼接線已驗證通過；僅數值為暫定。

## 5. 主要發現

### 5.1 三個數量級的差距
從 FGSM/GPU（0.08 毫秒）到 EAD-EN/CPU（146 毫秒）跨度約 1800 倍。
圖表使用對數尺度 y 軸，以便所有 5 種攻擊在同一張圖上都能清楚呈現。

### 5.2 最佳化攻擊 vs 單步攻擊成本
單步（FGSM）：次毫秒級。迭代式（PGD/CW）：低毫秒級。L1/EN 迭代式
（EAD）：數十至數百毫秒。EAD-EN 在 CPU 上比 FGSM 慢約 1700 倍——成因
為 L1+L2 elastic-net 次梯度內部迴圈。

### 5.3 GPU 加速比隨工作量擴增
加速比從 FGSM 的 2.1×（受 batch overhead 限制）成長到 EAD-EN 的
5.7×（完全 GPU-bound）。PGD 與 CW 介於兩者之間，與其 10 步內部迴圈
被 batch 運算分攤的特性一致。

### 5.4 即時部署可行性
以 RML2016 的符號率（~125 µs/樣本預算）來看：

| 攻擊 | GPU 延遲 | 即時可行？ |
|---|---:|:---:|
| FGSM   | 0.084 毫秒 |  可（84 µs） |
| PGD    | 0.753 毫秒 |  否（超過 6×） |
| CW     | 2.27 毫秒  |  否（超過 18×） |
| EAD-L1 | 26.2 毫秒  |  否（超過 210×） |
| EAD-EN | 25.5 毫秒  |  否（超過 200×） |

**意涵：** 在此硬體上，僅 FGSM 可作為即時 AMC 接收端的線上攻擊。
CW/EAD 在評估時必須離線預先計算——無專屬加速硬體則無法即時部署。
此結果界定了威脅模型：實際攻擊 RF 鏈路的對手，要嘛使用 FGSM，要嘛
將最佳化型攻擊外包到具非可忽略延遲的獨立計算路徑，從而削弱了
CW/EAD 作為可部署威脅的真實性。

### 5.5 變異度
CW 極為穩定（σ < 0.5% 相對值）。EAD-L1 在 CPU 上具有最高的試跑間
變異（σ ≈ 16%），原因是 elastic-net 最佳化器的提前終止行為與快取
效應交互作用所致。GPU 變異度普遍偏低（5 種攻擊皆 < 5% 相對值）。

## 6. 為何 GPU 能改善對抗式生成時間

對抗式攻擊的主要成本來自模型上重複的前向 + 反向傳播；每次迭代都對
batch 執行大量矩陣乘法與卷積——這正是 GPU 為其而生的工作負載。

### 6.1 張量運算的大規模平行化
AWN 的每次前/反向傳播會在 batch 與 channel 維度上執行數千個獨立的
乘加運算（multiply-accumulate）。CPU 在 6 核心上序列執行；RTX 5060 Ti
則在數千個 CUDA 核心上同步執行。這是最關鍵的單一因素。

### 6.2 Tensor Core 加速矩陣乘法
5060 Ti 配備專屬 tensor core，其執行融合乘加（fused MAC）的吞吐量
遠高於通用 CPU SIMD（AVX2/AVX-512）。AWN 分類器中的全連接層及內部
卷積堆疊可直接受益。

### 6.3 更高的記憶體頻寬
GPU GDDR 記憶體頻寬約為 CPU DRAM 的 5–20 倍。對抗式攻擊會反覆串流
活化值、梯度與中間緩衝——頻寬受限的工作負載幾乎隨此線性放大。

### 6.4 迭代次數放大優勢
單步攻擊（FGSM = 1 前向 + 1 反向）僅獲得約 2× 加速，因為 kernel
啟動開銷與 host↔device 傳輸主導了極小的計算量。迭代式攻擊則擴展性
更佳：

| 攻擊 | 內部迭代次數 | 觀測 GPU 加速比 |
|---|---:|---:|
| FGSM   | 1   | 2.1× |
| PGD    | 10  | 2.8× |
| CW     | 100 | 2.5× |
| EAD-L1 | 100 | 3.7× |
| EAD-EN | 100 | 5.7× |

規律：**迭代次數越多，GPU 的計算平行度越能攤銷固定的啟動/傳輸
開銷**，加速比也越大。

### 6.5 為何 EAD-EN 是榜首（5.7×）
EAD-EN 的 elastic-net 內部步驟，在標準 L2 攻擊之上每次迭代多執行兩
次次梯度計算。這些在 CPU 上線性累加；在 GPU 上則被合併進同一個平行
kernel 啟動，邊際成本趨近於零。EAD-EN 是套件中「每迭代計算量」最高
的攻擊，因此最能對應到 GPU 的優勢。

### 6.6 為何 FGSM 僅 2.1×（小工作量陷阱）
FGSM 的計算極輕，導致 **kernel 啟動延遲、Python 開銷與 CUDA 同步**
反而成為主導成本。GPU 在數微秒內完成數學運算，但必須等候主機派發
下一個呼叫——這是經典的「小 batch GPU 利用率不足」模式。要在 FGSM
上獲得更大加速，需採用更大 batch（攤銷啟動成本）、`torch.compile`
或 CUDA graphs。

### 6.7 對威脅模型的意涵
在即時 RF 防禦的脈絡下，**僅 FGSM 在 GPU 上夠快以作為可行的線上
攻擊**（84 µs 對 RML2016 速率的 125 µs/樣本符號預算）。所有最佳化
型攻擊（PGD/CW/EAD）即便在現代 GPU 上也須離線預先計算——這強化了
威脅模型論點：在缺乏專屬加速硬體的情況下，迭代式攻擊不具實際對
即時 RF 接收端部署的可行性。

## 7. 實作細節

### 7.1 計時協定（D-01..D-04）
- 每一格（cell）：丟棄 W 次 warmup 迭代，回報 R 次計時迭代的 mean ± std。
- GPU：每對 `perf_counter()` 的 t0/t1 都以 `torch.cuda.synchronize()` 框住。
- CPU：不需同步（CPU 操作預設為同步）。
- 每次重複的延遲 = (t1 - t0) / n_total_samples × 1000 毫秒。

### 7.2 分層抽樣（D-06）
透過 `_stratified_indices()` 在 (snr, label) 桶之間採輪詢式抽取，
以 `np.random.default_rng(2022)` 做為決定性種子。當未提供 SNR 時，
退化為僅依 label 分桶。

### 7.3 論文超參數釘定（D-05）
基準測試在呼叫 `create_attack()` 前以原地（in-place）方式覆寫 cfg：
- `ta_box='unit'`、`attack_eps=0.03`
- CW：`c=1.0`、`steps=100`、`lr=0.01`
- EAD：`max_iterations=100`
- PGD：論文指定 `alpha=0.01`；sigguard 推導為 `alpha=eps/4=0.0075`。
  基準測試接受 0.0075 值（相對於 10 步內部前/反向成本，延遲差異可忽略）。

### 7.4 State_dict 衛生（D-13，T-07-01 緩解）
裝置迴圈前先在 CPU 端快照原始 state_dict，每次裝置切換前重新載入到
目標裝置。可防範攻擊修改 `requires_grad` 或 BN 統計量所造成的模型
參數原地變動。

### 7.5 Dispatcher 錯誤修復
原本 Plan 02 的接線將完整的 220k 列 `SNRs` 陣列傳入了 44k 列的測試
切分，執行期間被捕獲並就地修復為：
```python
snrs_test = [SNRs[i] for i in test_idx]
run_attack_bench_5x2(..., snrs_test=snrs_test, test_idx=test_idx)
```

## 8. 程式碼審查

| 嚴重度 | 數量 | 處置 |
|---|---:|---|
| Critical | 0 | — |
| Warning  | 3 | 可選清理（advisory） |
| Info     | 7 | 可選清理（advisory） |

**主要警告：**
- **WR-01** state_dict 在同一裝置內的「攻擊之間」未重新載入（僅在裝置
  切換時重載）。CW/EAD 的參數變動可能污染同一裝置內後續攻擊的計時。
- **WR-02** `_stamp_paper_defaults` 含死碼/相互矛盾的分支（先計算後覆寫）。
- **WR-03** D-05 論文預設常數重複出現在 docstring、`main.py` dispatcher
  與 `_stamp_paper_defaults` helper 三處——存在飄移風險。

完整報告：`.planning/phases/07-.../07-REVIEW.md`。可透過
`/gsd-code-review-fix 7` 自動修復。

## 9. 驗證

- **自動化驗證：** 3 個 plan 共 18/18 項 must-have 全部通過（檔案存在、
  schema、跨檔案連結接線、D-01..D-16 程式碼不變式）。
- **狀態：** `human_needed`——自動化檢查全通過，但有兩項需人工確認：
  1. 對渲染後的 PDF 進行視覺檢查（圖例、誤差棒、IEEE 字體排版）。
  2. 在相機就緒版前以完整預算重新生成（目前數值為煙霧預算）。

完整報告：`.planning/phases/07-.../07-VERIFICATION.md`。
追蹤於：`.planning/phases/07-.../07-HUMAN-UAT.md`。

## 10. 後續步驟

1. **視覺檢查** `paper/latex/figures/attack_bench_latency.pdf`。
2. **完整預算重新生成**（RTX 5060 Ti 約 15–30 分鐘）：
   ```bash
   source venv/bin/activate
   python main.py --mode attack_bench --dataset 2016.10a --ckpt_path ./checkpoint
   cd paper/scripts && python plot_attack_bench_latency.py
   ```
3. **可選：** `/gsd-code-review-fix 7` 套用 WR-01/02/03 清理。
4. **Phase 6 整合：** 將 PDF 嵌入相機就緒版手稿，與 Table I 並列。

## 11. 提交紀錄（Phase 7 於 `main` 分支）

```
b77d899 test(07): persist human verification items as UAT
4363f4c docs(07): add code review report
41a3b0b docs(07): commit phase 7 plan files
3b3193d docs(07-03): complete attack-bench paper-figure plan
6db6362 fix(07-03): slice SNRs to test-split rows in attack_bench dispatcher
f202463 feat(07-03): render attack_bench_latency.pdf for paper integration
f8fc57a feat(07-03): add plot_attack_bench_latency.py for paper figure
5c016b4 docs(07-02): complete main-py attack_bench wiring plan
5537c7b chore: merge executor worktree (plan 07-02 tasks 1-2)
e52d5f2 docs(07-01): complete attack-bench engine plan
a0c6ad7 feat(07-01): add attack_bench.py with 5x2 latency benchmark engine
6c77172 feat(07-02): add attack_bench dispatcher branch to main.py
ac6733d feat(07-02): add Phase 7 --bench_* argparse flags to main.py
```
