#!/usr/bin/env python3
"""Randomized verification of tiled_gemm_s8 against numpy."""
import sys, os, subprocess, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iohex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AWN_DIR = os.path.join(SCRIPT_DIR, '..')
BUILD_DIR = os.path.join(AWN_DIR, 'build')


def compile_sim():
    os.makedirs(BUILD_DIR, exist_ok=True)
    sim = os.path.join(BUILD_DIR, 'sim_tiled_gemm_s8')
    subprocess.check_call([
        'iverilog', '-g2005-sv', '-o', sim,
        os.path.join(AWN_DIR, 'tb', 'tb_tiled_gemm_s8.v'),
        os.path.join(AWN_DIR, 'rtl', 'tiled_gemm_s8.v'),
        os.path.join(AWN_DIR, 'rtl', 'pe_s8.v'),
        os.path.join(AWN_DIR, 'rtl', 'bram_feeder_b.v'),
    ])
    return sim


def run_gemm(sim, M, K, N, A, B, bias=None):
    with tempfile.TemporaryDirectory() as tmp:
        a_f = os.path.join(tmp, 'a.hex')
        b_f = os.path.join(tmp, 'b.hex')
        o_f = os.path.join(tmp, 'out.hex')
        iohex.write_int8_hex(a_f, A.flatten())
        iohex.write_int8_hex(b_f, B.flatten())
        pa = [f'+M={M}', f'+K={K}', f'+N={N}',
              f'+a={a_f}', f'+b={b_f}', f'+out={o_f}']
        if bias is not None:
            bi_f = os.path.join(tmp, 'bi.hex')
            iohex.write_int32_hex(bi_f, bias.flatten())
            pa += ['+bias=1', f'+bi={bi_f}']
        else:
            pa += ['+bias=0']
        subprocess.check_call(['vvp', sim] + pa,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return iohex.read_int32_hex(o_f, count=M * N).reshape(M, N)


def test_one(sim, M, K, N, use_bias, label):
    A = np.random.randint(-128, 128, (M, K), dtype=np.int8)
    B = np.random.randint(-128, 128, (K, N), dtype=np.int8)
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

    # Single-tile (same range as systolic_mesh_s8)
    for M, K, N in [(1, 1, 1), (4, 4, 4), (8, 16, 16), (8, 320, 16)]:
        for ub in [False, True]:
            tid += 1
            ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')

    # Multi-tile aligned
    for M, K, N in [(16, 32, 32), (64, 320, 128)]:
        for ub in [False, True]:
            tid += 1
            ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')

    # M-boundary: last M-tile has 3 rows
    for ub in [False, True]:
        tid += 1
        ok &= test_one(sim, 11, 320, 1, ub, f'T{tid:02d}')

    # N-boundary: last N-tile has 2 cols
    for ub in [False, True]:
        tid += 1
        ok &= test_one(sim, 64, 192, 66, ub, f'T{tid:02d}')

    # Both M and N boundary
    for ub in [False, True]:
        tid += 1
        ok &= test_one(sim, 11, 192, 66, ub, f'T{tid:02d}')

    # Randomized tests
    for _ in range(20):
        M = int(np.random.choice([1, 2, 4, 7, 8, 9, 11, 16, 32, 64]))
        K = int(np.random.choice([1, 4, 16, 32, 64, 128, 192, 320]))
        N = int(np.random.choice([1, 2, 8, 15, 16, 17, 32, 64, 66, 128]))
        ub = bool(np.random.random() < 0.5)
        tid += 1
        ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')

    if ok:
        print(f'\nALL TILED SYSTOLIC TESTS PASSED ({tid} tests)')
        sys.exit(0)
    else:
        print(f'\nSOME TESTS FAILED')
        sys.exit(1)


if __name__ == '__main__':
    main()
