# CRC 實驗報告（基於 `crc_experiment.py`）

## 1. 報告摘要

本報告依據目前程式實作 `crc_experiment.py` 與輸出檔 `crc_experiment_results/crc_vs_amc.csv` 整理。  
核心結論是：**CW 攻擊先破壞 AMC 判識，再透過錯誤解調造成 CRC 大幅失敗**。在高 SNR（18 dB）下，若解調器知道正確 modulation（Oracle），CRC 仍大多維持高通過率；但若依賴被攻擊後的 AMC 輸出，CRC 幾乎崩潰。

## 2. 實驗目的

`crc_experiment.py` 目標是同時觀察兩件事：

- AMC 分類準確率（Track A，真實 RML2016.10a 資料）
- CRC 通過率（Track B，合成發射/接收鏈路）

並在四種場景下比較：

- `Clean+Oracle`
- `Clean+AMC`
- `Adv+Oracle`
- `Adv+AMC`

## 3. 方法與流程

### 3.1 Track A（AMC）

- 使用 `RML2016.10a` 子集做 AWN 分類，並產生 CW 對抗樣本。  
- 參考程式：`crc_experiment.py:73`, `crc_experiment.py:145`, `crc_experiment.py:185`

### 3.2 Track B（CRC）

- 以 `util.synth_txrx.generate_burst` 生成合成 burst，做 demod + CRC。  
- 只支援數位調變：`BPSK,QPSK,8PSK,QAM16,QAM64,PAM4,CPFSK,GFSK`。  
- 參考程式：`crc_experiment.py:50`, `crc_experiment.py:222`

### 3.3 對抗擾動轉移（Transfer）

- 先在 Track A 的真實資料上得到 `real_delta = adv_real - real_data`。
- 再將 `real_delta` 隨機加到 Track B 合成資料上。  
- 參考程式：`crc_experiment.py:212`, `crc_experiment.py:238`

### 3.4 四種場景定義

- `Clean+Oracle`：乾淨信號 + 已知真實 modulation 解調
- `Clean+AMC`：乾淨信號 + 用 AMC 結果（以 Track A 分佈模擬）解調
- `Adv+Oracle`：對抗信號 + 已知真實 modulation 解調
- `Adv+AMC`：對抗信號 + 用被攻擊後 AMC 結果解調

程式在對抗場景中會把 `adv_iq` 拼接回完整 burst 再解調，避免邊界 ISI 失真。  
參考程式：`crc_experiment.py:93`, `crc_experiment.py:283`

## 4. 資料覆蓋與輸出

- 結果檔：`crc_experiment_results/crc_vs_amc.csv`
- 共 88 筆（11 調變 × 2 SNR × 4 場景）
- 可做 CRC 的數位調變共有 64 筆（每 cell `n=200`）
- 類比調變 `WBFM, AM-DSB, AM-SSB` 為 `n=0`、`crc_pass=None`

## 5. 主要結果

### 5.1 數位調變整體平均（64 筆）

| Scenario | AMC Acc | CRC Pass |
|---|---:|---:|
| Clean+Oracle | 97.19% | 63.62% |
| Clean+AMC | 97.19% | 62.12% |
| Adv+Oracle | 4.65% | 59.59% |
| Adv+AMC | 4.65% | 4.12% |

重點：

- CW 攻擊把 AMC 從 `97.19%` 打到 `4.65%`（幾乎失效）
- 但在 Oracle 解調下，CRC 只從 `63.62%` 降到 `59.59%`
- 一旦用被攻擊 AMC 來解調（`Adv+AMC`），CRC 直接降到 `4.12%`

### 5.2 依 SNR 分層（數位調變平均）

| SNR | Scenario | AMC Acc | CRC Pass |
|---:|---|---:|---:|
| 0 dB | Clean+Oracle | 96.28% | 27.81% |
| 0 dB | Clean+AMC | 96.28% | 26.94% |
| 0 dB | Adv+Oracle | 2.30% | 25.94% |
| 0 dB | Adv+AMC | 2.30% | 0.44% |
| 18 dB | Clean+Oracle | 98.10% | 99.44% |
| 18 dB | Clean+AMC | 98.10% | 97.31% |
| 18 dB | Adv+Oracle | 7.00% | 93.25% |
| 18 dB | Adv+AMC | 7.00% | 7.81% |

重點：

- 在 `18 dB`，`Adv+Oracle` 仍有 `93.25%` CRC，顯示多數損害來自「錯解調」
- 在 `18 dB`，`Adv+AMC` 只剩 `7.81%` CRC，顯示 AMC 被攻擊是主要致命點
- 在 `0 dB`，基線 CRC 本來就低（`27.81%`），實驗結論更偏「系統在低 SNR 已脆弱」

### 5.3 `18 dB` 各調變 CRC（最能反映攻擊效果）

| Mod | Clean+Oracle | Adv+Oracle | Adv+AMC |
|---|---:|---:|---:|
| BPSK | 100.0% | 100.0% | 39.0% |
| QPSK | 100.0% | 100.0% | 1.5% |
| 8PSK | 100.0% | 100.0% | 0.5% |
| QAM16 | 100.0% | 99.5% | 0.5% |
| QAM64 | 96.0% | 76.0% | 0.5% |
| PAM4 | 100.0% | 71.0% | 2.0% |
| CPFSK | 100.0% | 100.0% | 18.5% |
| GFSK | 99.5% | 99.5% | 0.0% |

觀察：

- 多數調變在 `Adv+Oracle` 幾乎不掉（例如 BPSK/QPSK/8PSK/GFSK）
- `QAM64`、`PAM4` 在 `Adv+Oracle` 已明顯下滑，表示擾動本身也會破壞波形可解調性
- `Adv+AMC` 幾乎全面接近 0，說明錯誤 modulation 選擇是 CRC 崩潰主因

## 6. 結論

- 這個實驗較強地支持「**對抗攻擊可透過 AMC 誤判間接摧毀 CRC**」。
- 對於「擾動直接讓解調器失效」的證據較弱，主要出現在部分 modulation（如 `QAM64`, `PAM4`）。
- 因 Track B 為合成鏈路且擾動來自 Track A（transfer），結論屬於**可遷移性驗證**，不是完整真實鏈路端到端驗證。

## 7. 限制與風險

- Track B 並非真實 RML bit-level ground truth，而是合成 burst。
- `Clean+AMC` 在 Track B 不是直接跑 AMC，而是用 Track A 的準確率與錯分類抽樣來模擬。  
  參考程式：`crc_experiment.py:272`
- `0 dB` 下多數高階調變 Clean 基線 CRC 已很低，會放大或掩蓋攻擊效果。
- 類比調變無 CRC 評估（`n=0`）。

## 8. 建議下一步

- 新增 **Direct-on-TrackB**：直接以 CRC 或 demod loss 為目標，在合成 burst 上產生攻擊（不要只 transfer）。
- 固定並紀錄每次實驗設定（`cw_c`, `cw_steps`, `seed`, `n_bursts`）到結果檔中，提升可追溯性。
- 先聚焦 `18 dB` 做公平比較，再逐步擴展到更多 SNR。
- 針對 `QAM64`, `PAM4` 分析 `Adv+Oracle` 顯著下降原因（同步偏移、等化敏感度、功率正規化等）。
