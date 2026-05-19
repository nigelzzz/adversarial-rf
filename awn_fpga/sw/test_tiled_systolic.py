#!/usr/bin/env python3
"""Comprehensive verification of tiled_gemm_s8 against numpy.

Tests all 10 AWN GEMM shapes, boundary cases, 50 randomized dimensions,
and cross-checks against gemm_s8 behavioral reference when available.
"""
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


def compile_gemm_s8():
    """Compile behavioral gemm_s8 for cross-check. Returns None if unavailable."""
    tb_path = os.path.join(AWN_DIR, 'tb', 'tb_gemm_s8.v')
    rtl_path = os.path.join(AWN_DIR, 'rtl', 'gemm_s8.v')
    if not os.path.exists(tb_path) or not os.path.exists(rtl_path):
        return None
    os.makedirs(BUILD_DIR, exist_ok=True)
    sim = os.path.join(BUILD_DIR, 'sim_gemm_s8')
    try:
        subprocess.check_call([
            'iverilog', '-g2005-sv', '-o', sim,
            '-I', os.path.join(AWN_DIR, 'rtl'),
            tb_path,
        ], stderr=subprocess.DEVNULL)
        return sim
    except subprocess.CalledProcessError:
        return None


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

    # --- AWN GEMM shapes ---
    print('--- AWN GEMM shapes ---')
    awn_shapes = [
        (64,  14,  128, 'conv1'),
        (64,  320, 128, 'conv2'),
        (64,  192, 66,  'U.conv1'),
        (64,  192, 64,  'U.conv2'),
        (64,  192, 66,  'P.conv1'),
        (64,  192, 64,  'P.conv2'),
        (32,  128, 1,   'SE_lin0'),
        (128, 32,  1,   'SE_lin3'),
        (320, 128, 1,   'fc.0'),
        (11,  320, 1,   'fc.2'),
    ]
    for M, K, N, layer in awn_shapes:
        for ub in [False, True]:
            tid += 1
            ok &= test_one(sim, M, K, N, ub, f'T{tid:02d} ({layer})')

    # --- Boundary cases ---
    print('--- Boundary cases ---')
    boundary_cases = [
        (11,  320, 1,   'M+N boundary'),
        (64,  192, 66,  'N-boundary'),
        (32,  128, 1,   'SE linear'),
        (320, 128, 1,   'large M skinny N'),
        (1,   1,   1,   'minimum'),
        (8,   1,   16,  'min K full tile'),
        (9,   1,   17,  'M+N just over tile'),
    ]
    for M, K, N, desc in boundary_cases:
        for ub in [False, True]:
            tid += 1
            ok &= test_one(sim, M, K, N, ub, f'T{tid:02d} ({desc})')

    # --- 50 Randomized tests ---
    print('--- Randomized ---')
    M_choices = [1, 2, 3, 7, 8, 9, 11, 16, 32, 48, 64]
    K_choices = [1, 4, 14, 32, 64, 128, 192, 320]
    N_choices = [1, 2, 8, 15, 16, 17, 32, 64, 66, 128]
    for _ in range(50):
        M = int(np.random.choice(M_choices))
        K = int(np.random.choice(K_choices))
        N = int(np.random.choice(N_choices))
        ub = bool(np.random.random() < 0.5)
        tid += 1
        ok &= test_one(sim, M, K, N, ub, f'T{tid:02d}')

    # --- Cross-check vs gemm_s8 ---
    print('--- Cross-check vs gemm_s8 ---')
    gemm_sim = compile_gemm_s8()
    if gemm_sim is not None:
        M, K, N = 64, 320, 128
        A = np.random.randint(-128, 128, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 128, (K, N), dtype=np.int8)
        C_tiled = run_gemm(sim, M, K, N, A, B)
        C_behav = run_gemm(gemm_sim, M, K, N, A, B)
        xok = np.array_equal(C_tiled, C_behav)
        status = 'PASS' if xok else 'FAIL'
        print(f'{status}: (64,320,128) tiled output matches gemm_s8 byte-for-byte')
        ok &= xok
        tid += 1
    else:
        print('SKIP: tb_gemm_s8.v not found, cross-check skipped (numpy is authoritative)')

    # --- Summary ---
    if ok:
        print(f'\nALL TILED SYSTOLIC TESTS PASSED ({tid} tests)')
        sys.exit(0)
    else:
        print(f'\nSOME TESTS FAILED')
        sys.exit(1)


if __name__ == '__main__':
    main()
