# awn_hls — AWN INT8 in Vitis HLS 2023.2

Companion to the iverilog `awn_fpga/` flow. Reuses the same int8 golden
reference (`awn_fpga/build/quant.npz`) but reimplements the AWN forward pass
in Vitis HLS C++ so we can `csim` against golden and `csynth` to estimate
FPGA latency / resource cost.

## Layout

```
awn_hls/
  hello_world/       Phase 2: smoke test that vitis_hls works at all
  golden/            Phase 3: export quant.npz to C headers + hex testvectors
  src/               Phase 4: AWN INT8 HLS C++ port of sw/refmodel.py
  reports/           Phase 5: collected latency/resource tables
```

## Phase 1 — Install Vitis/Vivado 2023.2

You must run these steps interactively (AMD login + GUI installer).

### Disk and OS prerequisites already satisfied
- Ubuntu 22.04.3 LTS ✓
- 279 GB free on `/` ✓ (installer needs ~150 GB peak, ~120 GB after install)

### Install system libraries (one-time)
```bash
sudo apt update
sudo apt install -y \
    libtinfo5 libncurses5 libx11-6 libxext6 libxrender1 libxtst6 \
    libxi6 libgtk2.0-0 libcanberra-gtk-module libcanberra-gtk3-module \
    libnss3 libasound2 libxss1 libgbm1
```

### Installer
Web installer already downloaded: `~/FPGAs_AdaptiveSoCs_Unified_2023.2_1013_2256_Lin64.bin`

Run it:
```bash
chmod +x ~/FPGAs_AdaptiveSoCs_Unified_2023.2_1013_2256_Lin64.bin
~/FPGAs_AdaptiveSoCs_Unified_2023.2_1013_2256_Lin64.bin
```

The web installer needs internet access throughout; it downloads ~70 GB of
device files during install. Expect 1-3 hours depending on bandwidth.

### Installer GUI choices
- Product: **Vitis** (this auto-includes Vivado + Vitis HLS)
- Devices: select only the FPGA families you'll target
  (e.g. *Zynq UltraScale+ MPSoC* or *Versal AI Core* — skip the rest to save ~40 GB)
- Install path: `/tools/Xilinx` (default) or `~/Xilinx`
- Accept the cable-drivers EULA at the end

### After install — verify
```bash
source ~/vitas/Vitis/2023.2/settings64.sh
vitis_hls -version
vivado -version
```

Both commands should print `2023.2`. If yes → Phase 1 done.

(Install root on this machine: `~/vitas/` — yes, "vitas" not "vitis", typo
preserved from initial install.)

### Make the `source` permanent
Append to `~/.bashrc`:
```bash
echo 'source ~/vitas/Vitis/2023.2/settings64.sh > /dev/null 2>&1' >> ~/.bashrc
```

## Phase 2 — HLS hello-world sanity check

After Phase 1 succeeds:
```bash
cd awn_hls/hello_world
make csim       # C simulation only (fast)
make csynth     # C → RTL synthesis, produces latency + resource report
make clean      # remove build dirs
```

Expected:
```
*** C/RTL co-simulation finished: PASS ***
```
and a synthesis report under `hello_world/proj_simple_add/solution1/syn/report/`.

If this works, the toolchain is healthy and we proceed to Phase 3.

## Phase 3 — Golden reference (planned)

Will read `../build/quant.npz` and emit:
- `golden/awn_weights.h` — INT8 weights as C arrays
- `golden/awn_scales.h` — quantized multipliers + shifts
- `golden/golden_input.hex` — single IQ test vector (2×128 int8)
- `golden/golden_logits.txt` — expected fp32 logits + argmax

Source of truth is `sw/refmodel.py` (549 lines, all int8 forward arithmetic).

## Phase 4 — AWN HLS C++ (planned)

Port `sw/refmodel.py` ops to HLS:
- Conv2d / Conv1d → `ap_int<8>` MAC into `ap_int<32>` accumulator
- TFLite-style requantize using Q31 multiplier + right shift
- LeakyReLU / Tanh → LUT or piecewise-linear
- AdaptiveAvgPool1d / Linear / Concat

Target: csim output `argmax == 1` (AM-DSB) for `x_fp` golden input,
matching `fp_logits` reference.
