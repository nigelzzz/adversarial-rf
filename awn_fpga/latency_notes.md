# AWN FPGA — Latency 計算筆記

本文件說明如何估算 AWN 模型在 FPGA 上的推論 latency,從單一 op 的 cycle 公式一路推到端對端時間,並比較「拉時脈」與「加平行度」兩種優化路徑。

---

## 🎯 Latency 的三個層級

從低到高:

```
1. 單一 op latency           (一次 gemm 要幾個 cycle?)
2. 一張 sample 全網 latency   (34 個 op 串起來幾個 cycle?)
3. 端對端 latency / throughput  (含 DMA、batch、pipeline)
```

---

## 📐 第 1 層:單一 op 的 cycle 數

每個 RTL primitive 的 FSM 結構決定 cycle 公式。

### gemm_s8(最重要,佔 95%+ 計算)

看 `rtl/gemm_s8.v` 的狀態機:

```
S_IDLE  → S_INIT (1) → S_MAC (K cycles) → S_WRITE (1) → S_NEXT (1) → ...
```

每算一個輸出格子 `C[m,n]` 要:

```
INIT(1) + MAC(K) + WRITE(1) + NEXT(1) = K + 3 cycles
```

整個 GEMM 有 M·N 個輸出格子:

```
cycles(gemm) ≈ M · N · (K + 3) + 常數開銷
            ≈ M · N · K        (當 K >> 3)
            =  總 MAC 數
```

🔑 **關鍵**:這個 RTL 是「one-MAC-per-cycle」,所以 **cycle 數 ≈ MAC 數**。

### 各 primitive 的 cycle 公式

| Primitive | Cycle 公式 | 備註 |
|-----------|-----------|------|
| `gemm_s8` | M·N·K + O(M·N) | 一拍一個 MAC |
| `requantize_s32_s8` | LEN | per-element |
| `leaky_relu_s8` | LEN | per-element |
| `relu_s8` | LEN | per-element |
| `lut_s8`(tanh/sigmoid) | LEN | 查表,每元素 1 拍 |
| `eltwise_addsub_s8` | LEN | per-element |
| `mul_s8` | LEN | per-element |
| `avgpool1d_s8` | LEN + O(C) | 加總後 divide |

加上 FSM 啟動/結束開銷,通常每個 op 多 ~5 cycles。

---

## 📊 第 2 層:整網 latency(代真實數字)

### Step 1 — 列出 op 與 shape

從 `build/ops.json`:

| Op | Shape (M,K,N) | Cycles ≈ MACs |
|----|--------------|---------------|
| conv1 (Conv2d) | (64, 14, 128) | 114,688 |
| conv2 (Conv1d) | (64, 320, 128) | **2,621,440** |
| levels.U.op.1 | (64, 192, 66) | 811,008 |
| levels.U.op.4 | (64, 192, 64) | 786,432 |
| levels.P.op.1 | (64, 192, 66) | 811,008 |
| levels.P.op.4 | (64, 192, 64) | 786,432 |
| fc.0 (Linear) | (320, 128, 1) | 40,960 |
| SE Linear ×2 | — | 8,192 |
| fc.2 (Linear) | (11, 320, 1) | 3,520 |
| **GEMM 小計** | | **~5.98 M** |

### Step 2 — 加上非 GEMM op 的 cycle

非 GEMM op 的 cycle 由 **element 數**決定,不是 MAC:

| Op | Tensor size | Cycles |
|----|------------|--------|
| requantize ×8 | ~64·128 = 8,192 each | ~65,000 |
| leaky_relu ×5 | ~8,192 each | ~40,000 |
| tanh / sigmoid LUT ×3 | ~8,192 each | ~24,000 |
| eltwise add/sub ×2 | ~4,096 each | ~8,000 |
| avgpool1d ×2 | ~8,192 each | ~16,000 |
| mul ×1 | ~8,192 | ~8,000 |
| **非 GEMM 小計** | | **~161,000** |

### Step 3 — 總 cycle

```
總 cycles ≈ 5,983,680  (GEMM)
         +   161,000  (非 GEMM)
         +    34 × 10 (op FSM 啟動/結束開銷)
         ≈ 6,150,000 cycles
```

### Step 4 — 換算成時間

```
latency = cycles / f_clk

@ 100 MHz:  6.15M / 100M  ≈ 61.5 ms
@ 200 MHz:  6.15M / 200M  ≈ 30.8 ms
@ 50 MHz:   6.15M /  50M  ≈ 123 ms
```

🔑 **AWN 一張 sample 在現行 RTL 上 ~30–60 ms**,完全被 `gemm_s8`(43.8% MAC)主宰。

---

## ⚡ 第 3 層:端對端 latency(實務考量)

真實系統還要加:

```
total = T_dma_in + T_compute + T_dma_out + T_orchestrator
```

| 項目 | 估算 |
|------|------|
| **DMA in** (256 B IQ → onchip) | < 1 µs |
| **Compute** (上面算的 ~30 ms) | 主項 |
| **DMA out** (11 個 logit) | < 1 µs |
| **Orchestrator overhead** (CPU 排 op、param 設定) | 每 op ~10 µs × 34 ≈ 340 µs |

DMA 通常可以**跟 compute overlap**(ping-pong),所以實際 latency ≈ compute + orchestrator overhead。

---

## 🚀 如何降低 latency?(優化槓桿)

現在 `gemm_s8` 是 **1 MAC/cycle**,這是 latency 的根源。可以這樣加速:

### 方案 A:GEMM 平行化(最有效)

把 gemm 從 1 MAC/cycle 改成 **P MAC/cycle**:

| 平行度 P | conv2 cycles | 總 latency @100MHz |
|---------|-------------|-------------------|
| 1 (現在) | 2.62 M | 61.5 ms |
| 16 | 164 K | **3.8 ms** |
| 64 | 41 K | **0.96 ms** |
| 128 | 20.5 K | **0.5 ms** |

DSP 用量也會等比增加(P=64 約需 64 個 DSP48)。

### 方案 B:Pipeline / 跨 op overlap

讓 op#k 寫 C 的時候,op#k+1 就開始讀。這需要 double-buffer activation,可省 ~10–20%。

### 方案 C:N 維 unroll

`gemm_s8` 對固定 m 把 n 走過去 — N 維天生可 unroll(同 weight、不同 activation),`P_N=8` 很容易就拿到。

---

## 🧮 快速估算公式

```
T_inference  ≈  Σ MACs / (P · f_clk)  +  非GEMM_cycles / f_clk

代入:
  ΣMAC = 5.98 M
  非GEMM ≈ 0.16 M
  f_clk = 100 MHz

P=1   →  61.4 ms
P=16  →  4.0  ms
P=64  →  1.1  ms
```

---

## 🛠️ 怎麼實測 latency?

### 方法 1 — 從 testbench cycle count

看 iverilog testbench `$time` 差值:

```verilog
initial begin
    start_time = $time;
    @(posedge done);
    end_time = $time;
    $display("cycles = %0d", (end_time - start_time)/CLK_PERIOD);
end
```

### 方法 2 — 從 ops.json 估算(寫個 script)

```python
import json
ops = json.load(open('build/ops.json'))
total_cycles = sum(o.get('macs', 0) for o in ops)   # GEMM 主導
# 加上非 GEMM,粗估 *1.03 即可
T_ms = total_cycles * 1.03 / 100e6 * 1000
print(f"@100MHz, P=1: {T_ms:.1f} ms")
```

### 方法 3 — 上板實測

插一個自由跑的 32-bit counter,在 `start` 上升清零、`done` 上升鎖存,然後從 AXI-Lite 讀回。

---

## 🕒 拉時脈會降 latency 嗎?

**會,但有上限與代價。**

### 基本關係

```
latency (秒) = cycles / f_clk
```

cycles 不變的話,**f_clk 越高 → latency 越低**,呈線性關係。

```
@ 50  MHz:  6.15M / 50M  = 123 ms
@ 100 MHz:  6.15M / 100M =  61.5 ms   ← 加倍時脈,latency 砍半
@ 200 MHz:  6.15M / 200M =  30.8 ms
@ 400 MHz:  6.15M / 400M =  15.4 ms
```

### 限制 1:時脈不能無限拉(Timing Closure)

FPGA 跑得多快,由最長 combinational path 決定:

```
f_max = 1 / T_critical_path
```

看 `gemm_s8.v` 的關鍵路徑:

```verilog
acc <= acc + prod32;
//     ↑       ↑
//   讀 reg  8-bit × 8-bit 乘法器 + 32-bit 加法器
```

這條路有:
- BRAM 讀 a_buf, b_buf:~2 ns
- 8×8 乘法器:~3 ns(DSP48 約 2.5 ns)
- 32-bit 加法器:~2 ns
- Setup time:~0.5 ns
- **總共 ~7.5 ns → f_max ≈ 130 MHz**

要拉到 200 MHz 必須**插 pipeline register**:

```verilog
// 原本:1 個 cycle 做完讀+乘+累加
acc <= acc + (a * b);

// 改成:3 個 cycle 流水
stage1_prod <= a * b;             // 拍 1:乘法
stage2_prod <= stage1_prod;       // 拍 2:暫存
acc         <= acc + stage2_prod; // 拍 3:累加
```

代價:**MAC throughput 不變,但每個 op 的延遲多了幾拍**(對長 K 來說可忽略)。

### 限制 2:不同 FPGA 的時脈上限

| FPGA 系列 | 典型 BRAM Fmax | 典型 DSP Fmax |
|-----------|---------------|--------------|
| Cyclone IV(低階) | ~150 MHz | ~150 MHz |
| Artix-7 / Pynq-Z2 | ~250 MHz | ~300 MHz |
| Zynq UltraScale+ | ~500 MHz | ~600 MHz |
| Versal AI Edge | ~700 MHz+ | ~1 GHz(AI Engine) |

🔑 時脈不是免費資源,**板子等級決定上限**。

### 限制 3:功耗與時脈呈平方關係

```
P_dynamic ∝ C · V² · f
```

時脈 ×2:
- 動態功耗 ×2(同電壓)
- 但常常需要拉高 V_core 才穩定 → 功耗 ×2.5~3

對嵌入式板(Pynq-Z2 限 2W),拉時脈會撞功耗牆。

### 限制 4:對 DMA / 外部記憶體沒幫助

如果 latency 瓶頸在**搬資料**(weight 從 DDR 讀進來):

```
T_total = T_dma + T_compute
```

只有 `T_compute` 隨時脈縮,`T_dma` 由 **DDR 頻寬**決定,跟 fabric 時脈無關。

---

## 📊 拉時脈 vs 加平行度

兩個方法都能降 latency,但本質不同:

| 方法 | 改變什麼 | 上限 | 副作用 |
|------|---------|------|-------|
| **拉時脈 f_clk** | cycles 不變,每 cycle 更短 | timing closure (~300 MHz) | 功耗 ×2~3、需 pipeline 重寫 |
| **加平行度 P** | 每 cycle 做更多 MAC | DSP 資源耗盡 | 用更多 DSP/BRAM,功耗增加溫和 |

### 真實對照

以 AWN 為例(現行 1 MAC/cycle):

| 配置 | f_clk | P | Latency | DSP 用量 | 功耗估算 |
|------|-------|---|---------|---------|---------|
| 基準 | 100 MHz | 1 | 61 ms | 1 | 0.3 W |
| 純拉時脈 | 300 MHz | 1 | 20 ms | 1 | ~1 W |
| 純加平行 | 100 MHz | 64 | 0.96 ms | 64 | ~0.8 W |
| **組合** | 200 MHz | 32 | **0.96 ms** | 32 | ~1 W |

🔑 **加平行度的 latency 改善幅度遠超過拉時脈**,而且功耗效率更好。

---

## 🎯 實務優化建議

1. **先衝平行度**:`gemm_s8` 從 P=1 改到 P=16,改善 16 倍
2. **再優化 timing**:把可動的 f_clk 推到板子上限(Pynq-Z2 大概 200 MHz)
3. **最後配 pipeline**:把臨界路徑切細,讓 f_clk 還能再上去

組合起來:**P=16 @ 200 MHz → 1.9 ms**,比純拉時脈到 200 MHz(30 ms)快 16 倍。

---

## ✅ 總結

> **AWN FPGA latency 公式 = Σ MAC / (平行度 × 時脈)**。現行 RTL 是 1 MAC/cycle,在 100 MHz 下單張約 **60 ms**;只要把 `gemm_s8` 平行度開到 16,馬上掉到 **4 ms**,就能跑 real-time。
>
> **拉時脈會降 latency,但有 timing closure 上限(~300 MHz)和功耗代價**。對 AWN 這種計算密集模型,**加 GEMM 平行度比拉時脈更划算** — 同樣 16× 加速,平行度只多用 16 個 DSP,拉時脈卻要重做 pipeline 還燒兩倍功耗。
