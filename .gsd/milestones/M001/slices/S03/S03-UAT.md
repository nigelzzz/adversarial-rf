# S03: Hardware im2col Unit — UAT

**Milestone:** M001
**Written:** 2026-05-14T04:09:44.299Z

## UAT: S03 Hardware im2col Unit

### Gate 1: Full test suite passes
```bash
cd awn_fpga && python sw/test_im2col.py
```
**Expected**: `ALL IM2COL TESTS PASSED (68 tests)` with exit code 0.

### Gate 2: Individual config verification
```bash
cd awn_fpga && python sw/test_im2col.py --config D
cd awn_fpga && python sw/test_im2col.py --config B
cd awn_fpga && python sw/test_im2col.py --config C
cd awn_fpga && python sw/test_im2col.py --config A
```
**Expected**: Each prints PASS for all tests with no FAIL lines.

### Gate 3: Repo-root wrapper
```bash
python sw/test_im2col.py
```
**Expected**: Same 68-test PASS output as Gate 1.

### Gate 4: AWN conv1 shape (Config A critical path)
```bash
cd awn_fpga && python sw/test_im2col.py --config A 2>&1 | head -3
```
**Expected**: `PASS T01: cin=1 hin=2 win=134 kh=2 kw=7 -> K=14 N=128` (exact AWN conv1 im2col shape).
