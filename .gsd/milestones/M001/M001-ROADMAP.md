# M001: Systolic Array AWN Accelerator

## Vision
Replace the behavioral single-MAC gemm_s8 with an 8x16 output-stationary systolic array (128 PEs), add hardware im2col and AXI PS-PL interface, and prove end-to-end AWN inference latency <500 us at 200 MHz in iverilog simulation — meeting WiFi burst-level classification requirements.

## Slice Overview
| ID | Slice | Risk | Depends | Done | After this |
|----|-------|------|---------|------|------------|
| S01 | S01 | high | — | ⬜ | Single PE passes unit tests (MAC + accumulate + requantize); 8x16 mesh computes a small matrix multiply bit-exact against behavioral gemm_s8; randomized matrix tests pass |
| S02 | Tile Sequencer FSM + Weight Double-Buffering | high | S01 | ⬜ | Tile sequencer drives the 8x16 mesh through a full (64,320,128) GEMM with tiling, zero-padding boundary tiles, and weight double-buffering; output matches behavioral hex byte-for-byte |
| S03 | Hardware im2col Unit | medium | — | ⬜ | im2col FSM generates correct linearized addresses for all 3 kernel configs (2x7 conv1, 1x7 conv2, 1x17 lifting); output matches software im2col byte-for-byte via hex comparison |
| S04 | Full AWN Pipeline Controller | medium | S01, S02, S03 | ⬜ | All 38 ops execute through systolic GEMM + hardware im2col; non-GEMM modules unchanged and passing; per-layer cycle counts reported; end-to-end output matches behavioral refmodel byte-for-byte |
| S05 | AXI-Lite/DMA Interface + Latency Proof | low | S04 | ⬜ | PS can write IQ data via DMA, trigger inference via AXI-Lite, read back 11-class logits; total cycle count <100K (500 us at 200 MHz); simulation proves full PS-PL round-trip |
