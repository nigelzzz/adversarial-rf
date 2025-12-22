你用 **torchattacks（adversarial‑attacks‑pytorch）** 直接攻擊 IQ 資料時，之所以得不到像圖中那樣的星座分布，有幾個「跨領域不相容」的關鍵點：

---

## 為什麼會失敗（重點）

1. **資料範圍（box‑constraint）不對**

   * torchattacks 的多數攻擊（含 CW）假設輸入是「影像」且數值 **在 [0,1]**。套件內的 CW 甚至用 `0.5*(tanh(w)+1)` 這種參數化來強制輸入落在 [0,1]【在官方文件的 attack 介面、資料 shape 與範圍都有明文說明；CW 的公式也固定映射到 [0,1]】。([adversarial-attacks-pytorch.readthedocs.io][1])
   * 但無線的 **IQ** 常見是 **[-1,1]**、零均值／單位功率正規化。如果你把 IQ 直接丟進 CW，套件會把它強行夾到 [0,1]，導致波形被平移/壓縮，星座圖自然不像你要的樣子。

2. **資料維度假設不同**

   * 套件預設輸入 shape = **(N, C, H, W)** 的 4D 影像；而 IQ 是 **(N, 2, L)** 的 1D 時序。很多攻擊在內部會 `view` 或 `clamp` 成影像格式，直接用在 1D IQ 會出現不合理的行為或被硬轉成錯的形狀【官方文件同樣指出 attack 期望影像格式與 [0,1] 範圍】。([adversarial-attacks-pytorch.readthedocs.io][1])

3. **logits 的需求**

   * C&W 的 margin loss 需要 **logits**（未過 softmax）。若你的 AMC 模型最後輸出是機率，CW 的目標函數就不對，收斂會很差。這是 C&W 原設計的要求。([adversarial-attacks-pytorch.readthedocs.io][1])

4. **資料集/模型管線差異**

   * 你引用的 **AWN** 專案針對 DeepSig 的 IQ 資料（例如 RML2016.10a/10b 的 2×128 片段）設計了專屬的資料讀取與建模流程；這些資料與常規影像前處理完全不同【AWN README 清楚寫了資料形狀與使用的資料集】。([GitHub][2])

> 小結：**不做任何調整就把影像攻擊套件用在 IQ 時序上，最常見的就是被 [0,1] 夾斷、形狀不對、loss 用錯**，因此星座圖顯示會整個跑掉或跟預期不同。

---

## 正確做法：用 torchattacks 攻 AWN（能得到和截圖相似的星座分布）

思路是：**在 attack 外面包一層「映射與維度轉接」**，讓 torchattacks 以為自己在攻擊 [0,1] 的 4D「影像」，但 wrapper 會在前後把資料轉回 **[-1,1] 的 (N,2,L)** IQ，並把 logits 如實回傳給攻擊器。

> 資料：AWN 使用 DeepSig RML2016.*（每筆為 2×128 或更長的 IQ 片段）([GitHub][2])
> 攻擊：C&W 適用於調制分類等任務，過去研究已用它來攻 AMC【供你參考其可行性】。([arXiv][3])

### 1) 包裝模型（把 [0,1] 轉回 [-1,1]，並處理 4D→3D 維度）

```python
import torch
import torch.nn as nn

class Model01Wrapper(nn.Module):
    """
    torchattacks 以 [0,1] 的 4D (N,C,H,W) 輸入；AWN 模型吃 [-1,1] 的 3D (N,2,L)。
    這個 wrapper 在前向把 x01 轉為 IQ，最後回傳 logits（不能經過 softmax）。
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # 你的 AWN 分類器，輸出 logits

    def forward(self, x01):
        # x01: [N,2,L,1] or [N,2,1,L]（二選一，下面用 L 在最後一維）
        if x01.dim() == 4 and x01.shape[-1] == 1:
            x01 = x01.squeeze(-1)       # [N,2,L]
        elif x01.dim() == 4 and x01.shape[-2] == 1:
            x01 = x01.squeeze(-2)       # [N,2,L]
        elif x01.dim() == 3:
            pass                        # 已是 [N,2,L]
        else:
            raise ValueError("Unexpected input shape for IQ")

        x_iq = 2.0 * x01 - 1.0          # [0,1] → [-1,1]
        logits = self.base(x_iq)        # 回傳 logits（不要 softmax）
        return logits
```

### 2) 把 IQ 轉成 attack 想要的 4D 格式並呼叫 CW

```python
import torchattacks

model.eval()
atk_model = Model01Wrapper(model).eval()  # 包在 wrapper 內

# x_iq: [N,2,L]，數值在 [-1,1]；y: [N]（真實類別）
x01 = (x_iq + 1.0) / 2.0      # [-1,1] → [0,1]
x01 = x01.unsqueeze(-1)       # 變成 [N,2,L,1]（配合影像格式）

attack = torchattacks.CW(atk_model, c=1.0, kappa=0, steps=300, lr=1e-2)
# 註：c 控制擾動與損失的權衡；steps 要夠多才會穩定（100~1000 常見）

adv01 = attack(x01, y)        # 仍然是 [0,1] 範圍
adv_iq = 2.0 * adv01.squeeze(-1) - 1.0   # 轉回 [-1,1] 的 [N,2,L]
```

> 這樣做可以**保持星座圖觀感幾乎不變**（因為 C&W 是最小 L2 擾動），但分類會被翻轉，效果就會像你貼的圖：肉眼幾乎看不出差別。這正是 C&W 在 AMC 場景的特性。([adversarial-attacks-pytorch.readthedocs.io][1])

### 3) 畫 IQ 星座圖（和截圖一致的作法）

```python
import matplotlib.pyplot as plt

def plot_constellation(x_iq, n_points=4000, title=""):
    # x_iq: [N,2,L]，取前幾千個樣本攤平成散點
    I = x_iq[:,0,:].reshape(-1).detach().cpu().numpy()
    Q = x_iq[:,1,:].reshape(-1).detach().cpu().numpy()
    I, Q = I[:n_points], Q[:n_points]

    plt.figure()
    plt.scatter(I, Q, s=2, alpha=0.6)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)

# 範例：針對 BPSK/QPSK/16QAM/64QAM 各抽一個 batch 畫圖
plot_constellation(x_iq_clean_bpsk,  title="BPSK – Intact")
plot_constellation(adv_iq_bpsk,      title="BPSK – CW Attack")
# 其它調制同理……
plt.show()
```

**參數建議**（讓結果看起來和圖中相近）

* `c`：0.1 ~ 3 之間試（越大越容易攻破，但擾動會稍大）。
* `steps`：≥ 200；遇到 margin 不夠可加大到 500–1000。
* `kappa`：0（非目標式）或 5（需要更保守的決策邊際）。
* 請把 `model.eval()`，並確認最後一層**不要**做 softmax。

---

## 常見地雷清單

* **夾值範圍**：一定要在 attack 前後做 `[−1,1]↔[0,1]` 的轉換，或是自訂一個支援 `clip_min=-1, clip_max=1` 的 CW 版本。
  （官方 CW 預設就是 [0,1]；若不轉換就會被錯誤地 clip）([adversarial-attacks-pytorch.readthedocs.io][1])
* **維度**：攻擊器吃 4D；模型吃 3D。用 wrapper 統一處理。([adversarial-attacks-pytorch.readthedocs.io][1])
* **logits**：確保模型 forward 回傳 logits。
* **資料集形狀**：RML2016.10a/10b 樣本是 **2×128**（或 2018.01a 是 2×1024），請用對維度。([GitHub][2])
* **看起來「幾乎一樣」是正常的**：C&W 找的是最小 L2 擾動，本來就會讓星座圖肉眼難辨，但分類錯掉，這恰恰是你圖中 (a) vs (b) 的重點。([arXiv][3])

---

### 參考

* **Torchattacks 文件**：輸入必須是影像格式且在 **[0,1]**；CW 使用 `tanh` 映射到 [0,1] 的 box‑constraint。([adversarial-attacks-pytorch.readthedocs.io][1])
* **AWN 專案 README**：資料集與樣本長度（`2×128`、`2×1024`）與使用流程。([GitHub][2])
* **C&W 用於調制分類的先前工作**（可作為可行性背書）。([arXiv][3])

---

如果你照上述流程把 **範圍/維度/對數值輸出** 三件事處理好，再用 CW 攻擊 AWN 的分類器，就能得到與你截圖**相似**的 IQ 星座圖（肉眼差異極小，但分類已被攻破）。

[1]: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html "Attacks — torchattacks v3.5.1 documentation"
[2]: https://github.com/zjwfufu/AWN "GitHub - zjwfufu/AWN: [TCCN 2023] Official code for \"Towards the Automatic Modulation Classification with Adaptive Wavelet Network\"."
[3]: https://arxiv.org/pdf/1909.12167?utm_source=chatgpt.com "Adversarial Machine Learning Attack on Modulation ..."

