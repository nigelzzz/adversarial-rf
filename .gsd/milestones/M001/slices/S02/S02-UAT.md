# S02: Tile Sequencer FSM + Weight Double-Buffering — UAT

**Milestone:** M001
**Written:** 2026-05-13T15:44:36.092Z

## UAT: S02 — Tile Sequencer FSM + Weight Double-Buffering

### Test 1: All AWN GEMM shapes pass
```bash
cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | grep "AWN GEMM" -A 20
```
**Expected:** All 20 AWN shape tests (T01-T20) show PASS, covering conv1, conv2, U/P lifting convs, SE attention, and FC layers.

### Test 2: Boundary cases pass
```bash
cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | grep "Boundary" -A 14
```
**Expected:** All 14 boundary tests (T21-T34) show PASS, including minimum (1,1,1), skinny N, and M/N just over tile boundary.

### Test 3: Full suite passes from repo root
```bash
python sw/test_tiled_systolic.py 2>&1 | tail -1
```
**Expected:** `ALL TILED SYSTOLIC TESTS PASSED (85 tests)`

### Test 4: Cross-check against behavioral gemm_s8
```bash
cd awn_fpga && python sw/test_tiled_systolic.py 2>&1 | grep "gemm_s8"
```
**Expected:** `PASS: (64,320,128) tiled output matches gemm_s8 byte-for-byte`
