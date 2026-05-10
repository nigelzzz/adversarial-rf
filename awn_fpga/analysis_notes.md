# AWN FPGA — 分析筆記彙整

本文件彙整對 `awn_fpga/` 專案的幾個關鍵問答，內容包含：
1. Operator 命名解析（`levels.0.U.op.1` 等）
2. Lifting Scheme 理論（Updator / Predictor 是什麼）
3. README 重點翻譯（聚焦 Operator Profiling）
4. FPGA BRAM 用量估算
5. Wavelet block 是不是一個大 op

---

## 1. Operator 命名解析

### 1.1 README 排名前五的 Conv1d 是什麼

| 名次 | Op | Layer | MACs | 佔比 |
|---|---|---|---|---|
| 1 | Conv1d 64→64 k=5 | `conv2` | 2,621,440 | 43.8% |
| 2 | Conv1d 64→64 k=3 | `levels.0.U.op.1` | 811,008 | 13.6% |
| 3 | Conv1d 64→64 k=3 | `levels.0.P.op.1` | 811,008 | 13.6% |
| 4 | Conv1d 64→64 k=3 | `levels.0.U.op.4` | 786,432 | 13.1% |
| 5 | Conv1d 64→64 k=3 | `levels.0.P.op.4` | 786,432 | 13.1% |

> **`levels.0.*.op.*` 不是 SE-Attention**。SE-Attention 是後面那兩個 Linear（128→32→128）。

### 1.2 命名分段意義

完整路徑 `levels.0.U.op.1`：

| 段 | 意義 |
|---|---|
| `levels` | AWN 的 lifting 解碼層列表（`nn.ModuleList`） |
| `.0` | 第 0 層（這個 checkpoint 只用 1 層 lifting） |
| `.U` / `.P` | U = Updator 算子，P = Predictor 算子 |
| `.op` | Operator 內部的 `nn.Sequential` 容器 |
| `.1` / `.4` | Sequential 裡的第 1 個 / 第 4 個 sub-module |

### 1.3 Operator 內部結構（`lifting.py:25-40`）

每個 P 或 U 都是這個 Sequential：

```
op.0 : ReflectionPad1d(pad=2)   ← 純資料搬移，沒有 MAC
op.1 : Conv1d(64→64, k=3)       ← 第一個卷積  ← op.1
op.2 : LeakyReLU(0.01)
op.3 : Dropout
op.4 : Conv1d(64→64, k=3)       ← 第二個卷積  ← op.4
op.5 : Tanh
```

### 1.4 為什麼 op.1 跟 op.4 MAC 不同

兩個卷積規格相同（64→64, k=3），但中間沒再 pad，所以序列長度差 2：

```
輸入:           [B, 64, 64]    (lifting split 後的 odd 一半)
ReflPad(2):    [B, 64, 68]    ← 長度 +4
Conv1d k=3:    [B, 64, 66]    ← op.1: 64·64·3·66 = 811,008 MAC
LeakyReLU →    [B, 64, 66]
Conv1d k=3:    [B, 64, 64]    ← op.4: 64·64·3·64 = 786,432 MAC
Tanh →         [B, 64, 64]
```

### 1.5 lifting 為什麼佔 ~54% MAC

P + U 各跑一條 chain，每條兩個 Conv1d → 一層 4 個 Conv1d：

```
levels.0.U.op.1 + op.4  =  811,008 + 786,432  ≈ 1.60 M
levels.0.P.op.1 + op.4  =  811,008 + 786,432  ≈ 1.60 M
合計                    ≈ 3.19 M / 5.98 M     ≈ 53.4%
```

### 1.6 命名速查

```
conv1, conv2          ← 前端特徵抽取（CNN）
levels.0.U.op.1/4     ← lifting 的 Updator 兩個 Conv1d
levels.0.P.op.1/4     ← lifting 的 Predictor 兩個 Conv1d
SE_attention_score    ← SE-Attention 的兩個 Linear
fc.0, fc.2            ← 最後分類器
```

---

## 2. Lifting Scheme 理論

### 2.1 為什麼要做 wavelet

| 方法 | 看時間 | 看頻率 |
|---|---|---|
| 直接看 raw | ✓ | ✗ |
| FFT | ✗ | ✓ |
| Wavelet | ✓ | ✓（多尺度） |

對 modulation classification 特別有用：BPSK 跟 QAM64 的差別常常藏在不同時間尺度的頻率成分裡。

### 2.2 Lifting scheme 三步驟

```
        ┌── even ───────────────┐ (+) ──→ c (approx, 低頻)
input ──┤                       │
        │            ┌── U ────┤
        │            │
        └── odd  ────┴────────┐ (−) ──→ d (detail, 高頻)
                              │
                     ┌── P ───┘
                     │
                     c
```

1. **Split**：偶數位置 `even = x[0::2]`、奇數位置 `odd = x[1::2]`，長度各砍一半
2. **Update（U）**：`c = even + U(odd)` → 低頻、平滑
3. **Predict（P）**：`d = odd − P(c)` → 高頻、細節

對應 `lifting.py:69-70`：

```python
c = x_even + U(x_odd)
d = x_odd  − P(c)
```

### 2.3 Haar wavelet 範例

```
x    = [10, 12, 20, 18, 30, 34, 50, 48]
even = [10, 20, 30, 50]
odd  = [12, 18, 34, 48]

U(odd) = odd / 2
c = even + odd/2 = [16, 29, 47, 74]   ← 平均般的「平滑訊號」

P(c) = c
d = odd − c = [−4, −11, −13, −26]    ← 偏差量「細節訊號」
```

### 2.4 Updator vs Predictor 直觀分工

| 算子 | 直觀 | 做什麼 |
|---|---|---|
| **Updator U** | 「修正代表」 | 用 odd 補強 even，讓 c 變成更穩定的低頻平均 |
| **Predictor P** | 「猜 odd」 | 從 c 反推 odd 應該長怎樣，猜不到的殘差 = detail |

關鍵：**P 越強 → d 越小 → 訊號越平滑、能量越集中在低頻**。

### 2.5 AWN 的創新

傳統 wavelet（Haar / Daubechies）的 P、U 是寫死的多項式係數。
AWN **讓網路自己學 P 跟 U**，但用 Tanh 限制輸出範圍維持數值穩定：

```python
nn.Sequential(
    ReflectionPad1d(pad),
    Conv1d(64, 64, k=3),    # ← op.1
    LeakyReLU(0.01),
    Dropout(0),
    Conv1d(64, 64, k=3),    # ← op.4
    Tanh()                   # 限制輸出範圍
)
```

學出來的是「為了區分 11 個 modulation 客製化的小波」。這就是 **Adaptive Wavelet Network** 的由來。

---

## 3. README 重點（聚焦 Operator Profiling）

### 3.1 三個交付項

1. **Operator profile**：每個 op 的 shape、參數量、MAC
2. **Per-op iverilog 測試**：每個 op 對 numpy bit-exact 驗證
3. **End-to-end 推論**：完整 forward pass 過 iverilog，argmax 與 fp32 一致

### 3.2 TL;DR

```
Profiler:    34 ops, 5,983,680 MACs, 9 distinct hardware primitives
Op tests:    10 / 10 PASS  (every Verilog module bit-exact vs numpy ref)
Inference:   38 hw ops invoked, all bit-exact at every step
             argmax (int8 hw)  = class 1 (AM-DSB)
             argmax (fp32 ref) = class 1 (AM-DSB)
             argmax-match: TRUE
```

### 3.3 Profiler 工作流程（`sw/profile_awn.py`）

兩種互補方式抓 ops：

1. **`nn.Module` forward hook**：抓所有具名 sub-module
2. **Monkey-patch functional ops**：抓 `__add__`、`__sub__`、`torch.cat`、`torch.mul`，避免 lifting 內部 functional ops 漏算

輸出：
- `build/ops.json`（機器讀）
- `build/ops.txt`（人讀）

### 3.4 Op-kind 直方圖

```
LeakyReLU x5  ·  Conv1d x5  ·  Linear x4  ·  Dropout x3  ·  ReflectionPad1d x2
Tanh x2  ·  Add x2  ·  AdaptiveAvgPool1d x2
ZeroPad2d, Conv2d, BatchNorm1d, BatchNorm2d, Sub, Concat, ReLU, Sigmoid, Mul  各 x1
```

注意：**出現次數最多的不一定最貴**。Dropout 推論時是 identity；ReflectionPad 是純記憶體搬移；Conv1d 雖然只有 5 次卻吃掉 95%+ 的 MAC。

### 3.5 Lowering：34 ops → 9 hardware primitives

折疊規則：
- BatchNorm fold 進 Conv weight/bias
- Dropout / ReflectionPad / ZeroPad / Concat / Reshape 都是 sw data movement
- Conv 全部 lower 成 GEMM（im2col 在 sw 做）
- Tanh / Sigmoid 共用同一個 LUT module
- Add / Sub 共用 module，op_sel 切換

最終 9 個 primitive：

```
gemm_s8                  ← Conv1d / Conv2d / Linear 都靠這一個
requantize_s32_s8        ← int32 累加器 → int8（含跨層 rescale）
leaky_relu_s8            ← LeakyReLU
relu_s8                  ← SE 內部的 ReLU
lut_s8 (tanh)            ← lifting 算子尾端的 Tanh
lut_s8 (sigmoid)         ← SE-Attention 結尾
avgpool1d_s8             ← AdaptiveAvgPool1d
eltwise_addsub_s8        ← lifting 的 even+U(odd)、odd−P(c)
mul_s8                   ← SE-Attention 最後的 elementwise 相乘
```

### 3.6 為什麼這樣切分對硬體最划算

- 拿掉 lifting → 硬體少 4 個 primitive（LUT、eltwise add/sub、跨尺度 requant）
- 拿掉 SE-Attention → 少 1 個 LUT（sigmoid）+ 1 個 Mul
- 這個 trade-off 是後續優化的主要槓桿

---

## 4. FPGA BRAM 用量估算

### 4.1 模型實際儲存量

#### Weights（int8）

| Layer | int8 bytes |
|---|---|
| `conv1` Conv2d(1→64, k=2×7) | 896 |
| `conv2` Conv1d(64→64, k=5) | **20,480** |
| `levels.0.U.op.1` Conv1d k=3 | 12,288 |
| `levels.0.U.op.4` Conv1d k=3 | 12,288 |
| `levels.0.P.op.1` Conv1d k=3 | 12,288 |
| `levels.0.P.op.4` Conv1d k=3 | 12,288 |
| SE Linear 128→32 | 4,096 |
| SE Linear 32→128 | 4,096 |
| `fc.0` Linear 128→320 | **40,960** ← 最大 |
| `fc.2` Linear 320→11 | 3,520 |
| **Total weights** | **123,200 B ≈ 120 KB** |

加上 bias（int32）≈ 3.5 KB → 整包 **約 124 KB ≈ 0.99 Mbit**。

#### Activation peak

最大張量是 conv2 的 int32 累加器：64·128·4 = **32 KB**。

#### im2col 暫存（最吃 BRAM）

| 卷積 | M | K | N | A | B | C(int32) |
|---|---|---|---|---|---|---|
| conv1 | 128 | 14 | 64 | 1.8 KB | 0.9 KB | 32 KB |
| **conv2** | 128 | 320 | 64 | **40 KB** | 20 KB | **32 KB** |
| lifting Conv1d ×4 | ~64 | 192 | 64 | ~13 KB | 12 KB | ~17 KB |
| fc.0 | 1 | 128 | 320 | 128 B | **40 KB** | 1.25 KB |
| fc.2 | 1 | 320 | 11 | 320 B | 3.5 KB | 44 B |

**單一 op 瞬時 peak**：conv2 的 A(40K) + B(20K) + C(32K) ≈ **92 KB**。

### 4.2 RTL 現在寫死的 LEN

```
gemm_s8:    A_LEN=65536  B_LEN=65536  C_LEN=16384(int32)  BIAS_LEN=1024(int32)
relu/leaky/lut/mul/eltwise/requant: LEN=8192
avgpool1d:  LEN=16384
```

全部 instantiate 加總 ≈ **332 KB**（為 testbench 方便而過度配置）。

### 4.3 對應 FPGA BRAM 數（36 Kbit / block）

| 方案 | 容量 | 36Kb BRAM | 適合板子 |
|---|---|---|---|
| **A. 全部 weight onchip + 單 op peak activation** | ~212 KB | **~48** | Pynq-Z2 (140) 充裕 |
| **B. Weight 走 DDR streaming** | ~92 KB | **~21** | 連 Cyclone IV 都跑得動 |
| **C. RTL 現有 LEN 直接 synthesize** | ~332 KB | **~74** | Pynq-Z2 範圍內 |
| **D. C + 全部 weight onchip** | ~452 KB | **~100** | ZCU 系列輕鬆 |

### 4.4 推薦做法

1. 把 `gemm_s8` 的 LEN 降到實際需要（A_LEN/B_LEN ~49152、C_LEN ~8192 int32）
2. Weight 預載一塊獨立 ROM/BRAM（120 KB ≈ 27 BRAM）
3. Activation buffer ping-pong（兩塊 8 KB BRAM 切換）
4. LUT 256 entry × 8 bit 用 distributed RAM 即可

### 4.5 一句話結論

> **AWN int8 推論 BRAM 下限 ≈ 200 KB（~50 個 36Kb BRAM）**，由 weight 120 KB 跟 GEMM 工作集 ~90 KB 主導。實際 synthesize 前把 LEN 對齊模型 shape，可再減 30~40%。

---

## 5. Wavelet block 是不是一個大 op

**不是。** PyTorch 角度看像一塊，硬體實作是 **14~16 個 op 的子圖**。

### 5.1 PyTorch 視角：一個 module

```python
class LiftingScheme(nn.Module):
    def forward(self, x):
        x_even, x_odd = self.split(x)
        c = x_even + self.U(x_odd)
        d = x_odd  − self.P(c)
        return c, d
```

### 5.2 硬體視角：完整展開

```
split (sw, free)
│
├─ even ──→ requantize_s32_s8                         #1
│
├─ odd ──→ ReflectionPad1d (sw)
│         ├─ gemm_s8  (U.op.1)                        #2
│         ├─ requantize_s32_s8                        #3
│         ├─ leaky_relu_s8                            #4
│         ├─ gemm_s8  (U.op.4)                        #5
│         ├─ requantize_s32_s8                        #6
│         └─ lut_s8 (tanh)                            #7
│                       │
│                       ▼
│                eltwise_add_s8  (c = even + U(odd))   #8
│                       │
│                       ▼  (P chain on c)
│                ReflectionPad1d (sw)
│                ├─ gemm_s8  (P.op.1)                  #9
│                ├─ requantize_s32_s8                  #10
│                ├─ leaky_relu_s8                      #11
│                ├─ gemm_s8  (P.op.4)                  #12
│                ├─ requantize_s32_s8                  #13
│                └─ lut_s8 (tanh)                      #14
│                       │
│         odd ──→ requantize_s32_s8                    #15
│                       │
│                       ▼
│                eltwise_sub_s8  (d = odd − P(c))      #16
```

對照 README「38 hardware ops invoked」：**lifting 一層佔 16 / 38 ≈ 42% invocation 次數**，跟 ~54% MAC 一致。

### 5.3 為什麼不 fused 成單一 op

| 考量 | 結論 |
|---|---|
| **可重用性** | GEMM、requant、LUT、eltwise 也被 conv1/conv2、SE、fc 用到，包大顆還是要再實作一份 primitive |
| **驗證粒度** | 9 個 primitive 各自 bit-exact 比對 numpy 較好 debug；fused 出錯難定位 |
| **Quantization 控制** | 跨層 requant 需要 scale 對齊，切細才能在每個介面控 scale |
| **Pipeline 排程** | 切細之後 orchestrator 能跨 op 做 ping-pong / double buffer |

### 5.4 邏輯上仍可當作一塊

```python
import json
ops = json.load(open("build/ops.json"))
lifting_macs = sum(o["macs"] for o in ops if o["name"].startswith("levels.0"))
total_macs   = sum(o["macs"] for o in ops)
print(f"lifting share: {lifting_macs/total_macs:.1%}")
# → lifting share: 53.4%
```

### 5.5 一句話結論

> **概念上是一塊（一層 lifting = 一次 wavelet 分解），實作上是 14~16 個 primitive 的 DAG**。AWN 切細是刻意的，這樣 9 個 RTL primitive 就涵蓋整個網路所有運算，不需要為 lifting 單獨做 fused IP。

---

## 6. 總結對照表

| 問題 | 答案 |
|---|---|
| `levels.0.U.op.1/4` 是什麼？ | Lifting 的 Updator 內 Sequential 的兩個 Conv1d（不是 SE-Attention） |
| `levels.0.P.op.1/4` 是什麼？ | Lifting 的 Predictor 內 Sequential 的兩個 Conv1d |
| Lifting 是什麼？ | 把訊號 split 成 even/odd，用 U 算低頻 c、用 P 算殘差 d 的小波分解食譜 |
| AWN 創新是什麼？ | 把 P/U 換成可學 CNN，自動發明適合 modulation 分類的小波 |
| Lifting 佔多少計算？ | ~54% MAC、42% op invocation |
| FPGA BRAM 需要多少？ | 下限 ~200 KB（~50 個 36Kb BRAM）；目前 RTL LEN ~332 KB |
| Wavelet 是一個大 op 嗎？ | 不是；硬體展開 14~16 個 primitive 的子圖 |
| 9 個硬體 primitive 是哪些？ | gemm、requant、leaky_relu、relu、lut(tanh)、lut(sigmoid)、avgpool1d、eltwise_addsub、mul |
