# AMC-Oriented CW Adversarial Attack 對 CRC 完整性的影響：以 `crc_experiment.py` 為例

## Abstract
本研究評估一種兩階段通訊接收流程在對抗擾動下的脆弱性：先做自動調變分類（AMC），再依預測調變進行解調與 CRC 驗證。實驗採用 `crc_experiment.py` 的雙軌設計。Track A 使用真實 `RML2016.10a` IQ 資料與 AWN 分類器，建立 clean 與 Carlini-Wagner（CW）攻擊下的 AMC 行為；Track B 使用合成 burst（具已知 bit 與 CRC）量測資料完整性。攻擊擾動先在 Track A 生成，再轉移（transfer）到 Track B。結果顯示：在可做 CRC 的數位調變集合（8 種 modulation）上，平均 AMC 準確率由 clean 的 97.19% 降至 adversarial 的 4.65%。然而，若解調端使用 oracle modulation，CRC 通過率仍由 63.62% 僅降至 59.59%；一旦改用被攻擊 AMC 輸出（Adv+AMC），CRC 通過率進一步降至 4.12%。在 18 dB 條件下，此現象更明顯：Adv+Oracle CRC 平均 93.25%，Adv+AMC 僅 7.81%。本研究指出，主要風險路徑是「AMC 誤判導致錯誤解調」，而非所有 modulation 都會在正確解調下直接失效。

## 1. Introduction
實務接收機常採用「先判識 modulation，再選擇對應解調器」的管線。此設計對抗噪聲有效，但若前段 AMC 被對抗擾動誤導，後段可能選錯 demodulation mode，最終造成 bit error 與 CRC failure。本研究關注以下問題：

1. CW 擾動主要破壞的是 AMC 還是波形可解調性本身？
2. 在有/無 oracle modulation 的情境下，CRC 表現差異有多大？
3. 在不同 SNR（0 與 18 dB）下，系統脆弱點是否一致？

## 2. Method

### 2.1 Two-Track 設計
實驗依 `crc_experiment.py` 分為兩軌：

1. Track A（AMC）：以 `RML2016.10a` 真實樣本評估 AWN，在 clean 與 CW 下計算分類準確率，並取得對抗擾動 `delta`。  
   對應程式：`crc_experiment.py:73`, `crc_experiment.py:175`, `crc_experiment.py:185`, `crc_experiment.py:212`
2. Track B（CRC）：以 `util.synth_txrx.generate_burst` 產生可重建 bit 的合成 burst，執行解調並計算 CRC 通過率。  
   對應程式：`crc_experiment.py:222`, `crc_experiment.py:228`

### 2.2 擾動轉移（Transfer Perturbation）
在 Track A 取得 `real_delta = adv_real - real_data` 後，隨機抽樣加到 Track B 合成 IQ：

\[
x^{\text{adv}}_{\text{synth}} = x_{\text{synth}} + \delta_{\text{real}}
\]

對應程式：`crc_experiment.py:238`, `crc_experiment.py:241`。

### 2.3 四種評估場景
對每個 `(mod, snr)` cell，評估四種場景：

1. `Clean+Oracle`：clean IQ + 真值 modulation 解調
2. `Clean+AMC`：clean IQ + 依 AMC 結果解調（以 Track A 分佈模擬）
3. `Adv+Oracle`：adversarial IQ + 真值 modulation 解調
4. `Adv+AMC`：adversarial IQ + 被攻擊後 AMC 結果解調

對應程式：`crc_experiment.py:268` 到 `crc_experiment.py:296`。  
其中對抗樣本在解調前會拼回完整 burst 以保留 guard context，減少邊界 ISI 偏差。對應程式：`crc_experiment.py:93` 到 `crc_experiment.py:124`。

## 3. Experimental Setup

### 3.1 模型、資料與調變
1. AMC 模型：AWN（由 `2016.10a_AWN.pkl` 載入）  
   對應程式：`crc_experiment.py:57`, `crc_experiment.py:63`
2. 真實資料：`./data/RML2016.10a_dict.pkl`（Track A）
3. 合成資料：`generate_burst`（Track B）
4. 可做 CRC 的 modulation：`BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4, CPFSK, GFSK`  
   對應程式：`crc_experiment.py:50`

### 3.2 攻擊設定與評估範圍
1. 攻擊：CW（由 `create_attack('cw', ...)` 建立）  
   對應程式：`crc_experiment.py:160`, `crc_experiment.py:161`
2. 每個 Track A cell 取 500 筆真實樣本（`n_samples=500`）  
   對應程式：`crc_experiment.py:175`
3. 本次結果檔顯示每個 Track B cell 為 `n=200`（數位 modulation）
4. SNR：0 與 18 dB（由結果檔反映）

資料來源檔案：`crc_experiment_results/crc_vs_amc.csv`。

## 4. Results

### 4.1 數位 modulation 整體平均（64 筆）

| Scenario | AMC Acc | CRC Pass |
|---|---:|---:|
| Clean+Oracle | 97.19% | 63.62% |
| Clean+AMC | 97.19% | 62.12% |
| Adv+Oracle | 4.65% | 59.59% |
| Adv+AMC | 4.65% | 4.12% |

關鍵觀察：CW 幾乎摧毀 AMC；但若使用 oracle modulation，CRC 平均僅小幅下降。CRC 的主崩潰發生在「被攻擊 AMC 驅動解調」場景。

### 4.2 SNR 分層結果

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

18 dB 下的對比最具代表性：`Adv+Oracle` 與 `Adv+AMC` 之間有 85.44 個百分點落差（93.25% vs 7.81%）。

### 4.3 18 dB 各 modulation CRC

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

補充統計：
1. 在 16 個 `(mod,snr)` 數位 cell 中，有 13 個 cell 的 `Clean+Oracle` 與 `Adv+Oracle` 差距在 2 個百分點內。
2. 有 14 個 cell 的 `Adv+AMC` CRC 不超過 5%。

## 5. Discussion

### 5.1 主要攻擊路徑：分類誤導而非全面波形毀損
結果一致顯示：CW 對 AMC 極具破壞力，但在 oracle 解調下，許多 modulation 的 CRC 仍維持高值。這表示攻擊對系統 end-to-end 的最大傷害，多數來自「前段分類誤導 -> 後段選錯解調器」。

### 5.2 為何 18 dB 更能凸顯機制
0 dB 下部分 modulation 的 clean CRC 已偏低（例如 8PSK/QAM16/QAM64），使攻擊效果與本底失敗混雜。18 dB 下 clean baseline 幾乎飽和，更能清楚量到攻擊導致的額外退化。

### 5.3 modulation 間差異
`QAM64` 與 `PAM4` 在 `Adv+Oracle` 仍有顯著下降（96.0%->76.0%，100.0%->71.0%），顯示除分類誤導外，擾動本身也可能破壞高階星座的解調穩定性。此現象值得在同步、等化與功率正規化層面做更細緻診斷。

## 6. Limitations and Validity Threats
1. 本研究為 transfer 設計：擾動生成於 Track A（真實資料），作用於 Track B（合成資料）；結論偏向「可遷移性」而非完整實網端到端攻擊。
2. `Clean+AMC`/`Adv+AMC` 在 Track B 並非逐筆跑 AMC，而是用 Track A 的分類結果分佈進行模擬。  
   對應程式：`crc_experiment.py:272`, `crc_experiment.py:289`
3. 結果檔未記錄完整攻擊超參數（如 `cw_c`, `cw_steps`, `ta_box`），重現需回看執行命令或腳本預設值。
4. 類比 modulation（`WBFM`, `AM-DSB`, `AM-SSB`）無 CRC 指標（`n=0`）。

## 7. Conclusion
在本實驗設定下，CW 攻擊對 AMC 造成近乎致命的誤導，並透過錯誤解調路徑使 CRC 通過率大幅下滑。關鍵訊息是：**系統風險集中在「分類依賴型接收流程」**，而不只是單純的波形失真。後續若要提升外部效度，建議加入 Direct-on-TrackB（以 demod/CRC 目標直接優化）與真實 bit-level 鏈路驗證。

## Reproducibility Note
本稿數值來自 `crc_experiment_results/crc_vs_amc.csv`。若需重跑，入口指令為：

```bash
python crc_experiment.py --ckpt_path ./checkpoint --output_dir ./crc_experiment_results
```

建議額外固定並明確記錄 `--n_bursts`, `--cw_c`, `--cw_steps`, `--seed`, `--ta_box` 以確保可重現性。
