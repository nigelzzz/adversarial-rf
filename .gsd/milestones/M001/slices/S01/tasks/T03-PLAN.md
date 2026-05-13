---
estimated_steps: 187
estimated_files: 2
skills_used: []
---

# T03: Create mesh testbench and randomized verification

## Description

Create the mesh testbench and a Python randomized verification script that proves the systolic array produces bit-exact int32 results matching numpy for all single-tile matrix sizes (M≤8, N≤16, arbitrary K). The testbench follows the existing `tb_gemm_s8.v` pattern exactly (plusargs, $readmemh, $fwrite). The Python script generates random int8 matrices, writes hex vectors, runs the iverilog simulation, and asserts zero divergence.

## Steps

1. Read `awn_fpga/tb/tb_gemm_s8.v` for the testbench pattern. Key elements:
   - Plusargs: `$value$plusargs("M=%d", M_arg)`, similarly K, N, bias, a (path), b (path), bi (path), out (path)
   - Data loading: `$readmemh(a_path, DUT.a_buf)` — loads hex file directly into DUT's internal array via hierarchy
   - Output dump: `$fwrite(fout, "%08x\n", DUT.c_buf[k] & 32'hffffffff)` — 8-char hex, one int32 per line
   - Clock: 20ns period (10ns half)
   - Reset: rst_n=0 → wait → rst_n=1 → negedge → start=1 → negedge → start=0
   - Timeout: based on `8 * M * K * N + 100000` cycles

2. Read `awn_fpga/sw/iohex.py` for hex I/O helpers:
   - `write_int8_hex(path, arr)`: writes numpy int8 array as 2-char hex bytes, one per line
   - `write_int32_hex(path, arr)`: writes numpy int32 array as 8-char hex words, one per line
   - `read_int32_hex(path, count)`: reads hex words back to int32 array

3. Create `awn_fpga/tb/tb_systolic_mesh_s8.v`:

```verilog
module tb_systolic_mesh_s8;
    reg clk = 0;
    always #10 clk = ~clk;   // 20ns period
    
    reg rst_n, start;
    reg [15:0] M_arg, K_arg, N_arg;
    reg bias_arg;
    wire done;
    
    systolic_mesh_s8 DUT (
        .clk(clk), .rst_n(rst_n),
        .start(start),
        .M_in(M_arg), .K_in(K_arg), .N_in(N_arg),
        .use_bias(bias_arg),
        .done(done)
    );
    
    reg [256*8-1:0] a_path, b_path, bias_path, out_path;
    integer fout, k;
    
    initial begin
        $value$plusargs("M=%d", M_arg);
        $value$plusargs("K=%d", K_arg);
        $value$plusargs("N=%d", N_arg);
        $value$plusargs("bias=%d", bias_arg);
        $value$plusargs("a=%s", a_path);
        $value$plusargs("b=%s", b_path);
        if (bias_arg) $value$plusargs("bi=%s", bias_path);
        $value$plusargs("out=%s", out_path);
        
        $readmemh(a_path, DUT.a_buf);
        $readmemh(b_path, DUT.b_buf);
        if (bias_arg) $readmemh(bias_path, DUT.bias_buf);
        
        rst_n = 0; start = 0;
        #40;
        rst_n = 1;
        @(negedge clk);
        start = 1;
        @(negedge clk);
        start = 0;
        
        wait (done);
        @(negedge clk);
        
        fout = $fopen(out_path, "w");
        for (k = 0; k < M_arg * N_arg; k = k + 1)
            $fwrite(fout, "%08x\n", DUT.c_buf[k] & 32'hffffffff);
        $fclose(fout);
        $finish;
    end
    
    // Timeout
    initial begin
        #(20 * (8 * M_arg * K_arg * N_arg + 200000));
        $display("TIMEOUT at cycle %0d", $time/20);
        $finish;
    end
endmodule
```

**iverilog notes:** Declare loop variable `k` as `integer k` at module level (not inside for-loop). Use `$value$plusargs` return value or separate `initial begin` block for bias_arg check. If `if (bias_arg)` before `$readmemh` causes issues, use a conditional: always call `$value$plusargs` but only pass to `$readmemh` when bias_arg is set.

4. Create `awn_fpga/sw/test_systolic.py` — randomized verification script:

```python
#!/usr/bin/env python3
"""Randomized verification of systolic_mesh_s8 against numpy."""
import sys, os, subprocess, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iohex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AWN_DIR = os.path.join(SCRIPT_DIR, '..')
BUILD_DIR = os.path.join(AWN_DIR, 'build')

def compile_sim():
    os.makedirs(BUILD_DIR, exist_ok=True)
    sim = os.path.join(BUILD_DIR, 'sim_systolic_mesh_s8')
    subprocess.check_call([
        'iverilog', '-g2005-sv', '-o', sim,
        os.path.join(AWN_DIR, 'tb', 'tb_systolic_mesh_s8.v'),
        os.path.join(AWN_DIR, 'rtl', 'systolic_mesh_s8.v'),
        os.path.join(AWN_DIR, 'rtl', 'pe_s8.v'),
    ])
    return sim

def run_gemm(sim, M, K, N, A, B, bias=None):
    with tempfile.TemporaryDirectory() as tmp:
        a_f = os.path.join(tmp, 'a.hex')
        b_f = os.path.join(tmp, 'b.hex')
        o_f = os.path.join(tmp, 'out.hex')
        iohex.write_int8_hex(a_f, A.flatten())
        iohex.write_int8_hex(b_f, B.flatten())
        pa = [f'+M={M}', f'+K={K}', f'+N={N}', f'+a={a_f}', f'+b={b_f}', f'+out={o_f}']
        if bias is not None:
            bi_f = os.path.join(tmp, 'bi.hex')
            iohex.write_int32_hex(bi_f, bias.flatten())
            pa += ['+bias=1', f'+bi={bi_f}']
        else:
            pa += ['+bias=0']
        subprocess.check_call(['vvp', sim] + pa, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return iohex.read_int32_hex(o_f, count=M*N).reshape(M, N)

def test_one(sim, M, K, N, use_bias, label):
    A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
    B = np.random.randint(-128, 127, (K, N), dtype=np.int8)
    bias = np.random.randint(-100000, 100000, (M,), dtype=np.int32) if use_bias else None
    C_ref = A.astype(np.int32) @ B.astype(np.int32)
    if bias is not None:
        C_ref += bias[:, np.newaxis]
    C_hw = run_gemm(sim, M, K, N, A, B, bias)
    ok = np.array_equal(C_ref, C_hw)
    status = 'PASS' if ok else 'FAIL'
    print(f'{status} {label}: M={M} K={K} N={N} bias={use_bias}')
    if not ok:
        diffs = np.argwhere(C_ref != C_hw)
        for idx in diffs[:5]:
            print(f'  C[{idx[0]},{idx[1]}]: expected {C_ref[tuple(idx)]}, got {C_hw[tuple(idx)]}')
    return ok

def main():
    np.random.seed(42)
    sim = compile_sim()
    ok = True
    tid = 0
    # Deterministic tests
    for M, K, N in [(1,1,1), (2,2,2), (4,4,4), (8,16,16), (8,1,16), (1,64,1), (8,320,16)]:
        for ub in [False, True]:
            tid += 1
            ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')
    # Randomized tests
    for _ in range(50):
        M = int(np.random.choice([1, 2, 4, 7, 8]))
        K = int(np.random.choice([1, 4, 16, 32, 64, 128, 192, 320]))
        N = int(np.random.choice([1, 2, 8, 15, 16]))
        ub = bool(np.random.random() < 0.5)
        tid += 1
        ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')
    if ok:
        print(f'\nALL SYSTOLIC TESTS PASSED ({tid} tests)')
        sys.exit(0)
    else:
        print(f'\nSOME TESTS FAILED')
        sys.exit(1)

if __name__ == '__main__':
    main()
```

**Key details:**
- Use `iohex.write_int8_hex` / `iohex.write_int32_hex` / `iohex.read_int32_hex` — do NOT rewrite hex I/O
- A is stored row-major: A[m,k] at flat index m*K+k. B is row-major: B[k,n] at flat index k*N+n. This matches gemm_s8.v's indexing.
- numpy int8 range: `np.random.randint(-128, 127)` gives -128..126. For full range use `randint(-128, 128)` which gives -128..127.
- Reference: `C = A.astype(np.int32) @ B.astype(np.int32)` — cast BEFORE matmul to get int32 arithmetic (numpy's int8 @ int8 would overflow)
- Bias: broadcast `bias[m]` across all N columns: `C_ref += bias[:, np.newaxis]`
- Test sizes: M∈{1,2,4,7,8}, K∈{1..320}, N∈{1,2,8,15,16} — all within single-tile limits (M≤8, N≤16)
- The 8,320,16 test exercises the largest K dimension used in the AWN pipeline (conv2 K=320)

5. Run full verification:
```bash
cd awn_fpga && python sw/test_systolic.py
```

Expected: 62+ tests (14 deterministic + 50 randomized), all PASS, exit code 0.

## Must-Haves

- [ ] Testbench uses same plusargs as tb_gemm_s8.v (M, K, N, bias, a, b, bi, out)
- [ ] Testbench loads data via $readmemh into DUT.a_buf, DUT.b_buf, DUT.bias_buf
- [ ] Testbench outputs via $fwrite with %08x format, masked with 32'hffffffff
- [ ] Python script uses iohex.py helpers (no custom hex I/O)
- [ ] 50+ randomized tests covering M∈{1..8}, K∈{1..320}, N∈{1..16}
- [ ] Tests with and without bias
- [ ] All tests bit-exact match: `np.array_equal(C_ref, C_hw)`
- [ ] Script exits 0 and prints "ALL SYSTOLIC TESTS PASSED" on success

## Verification

- `cd awn_fpga && python sw/test_systolic.py` exits 0 and prints "ALL SYSTOLIC TESTS PASSED"

## Negative Tests

- **Degenerate**: M=1, K=1, N=1 (single multiply, no accumulation)
- **Column vector**: M=8, K=320, N=1 (15 of 16 columns idle)
- **Row vector**: M=1, K=64, N=16 (7 of 8 rows idle)
- **Max pipeline depth**: M=8, K=320, N=16 (K+PM+PN-2 = 342 compute cycles)
- **Signed extremes**: covered by randomized tests with np.random.randint(-128, 127)

## Inputs

- ``awn_fpga/rtl/systolic_mesh_s8.v` — mesh module (created in T02)`
- ``awn_fpga/rtl/pe_s8.v` — PE module (created in T01)`
- ``awn_fpga/tb/tb_gemm_s8.v` — testbench pattern reference (plusargs, hex I/O, reset sequence)`
- ``awn_fpga/sw/iohex.py` — hex I/O helpers (write_int8_hex, write_int32_hex, read_int32_hex)`

## Expected Output

- ``awn_fpga/tb/tb_systolic_mesh_s8.v` — mesh testbench with plusargs interface matching tb_gemm_s8.v`
- ``awn_fpga/sw/test_systolic.py` — randomized verification script (62+ tests, bit-exact comparison against numpy)`

## Verification

cd awn_fpga && python sw/test_systolic.py
