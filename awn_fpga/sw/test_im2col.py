#!/usr/bin/env python3
"""Verification of im2col_addr_gen against numpy golden reference."""
import sys, os, subprocess, tempfile, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import iohex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AWN_DIR = os.path.join(SCRIPT_DIR, '..')
BUILD_DIR = os.path.join(AWN_DIR, 'build')


def compile_sim():
    os.makedirs(BUILD_DIR, exist_ok=True)
    sim = os.path.join(BUILD_DIR, 'sim_im2col')
    subprocess.check_call([
        'iverilog', '-g2005-sv', '-o', sim,
        os.path.join(AWN_DIR, 'tb', 'tb_im2col_addr_gen.v'),
        os.path.join(AWN_DIR, 'rtl', 'im2col_addr_gen.v'),
    ])
    return sim


def im2col_1d_ref(fmap_2d, kw, pad_left, pad_mode):
    """fmap_2d: (Cin, Win) int8. Returns (Cin*kw, Nout) int8."""
    cin, win = fmap_2d.shape
    if pad_mode == 0:
        xp = fmap_2d
    elif pad_mode == 1:
        xp = np.pad(fmap_2d, ((0, 0), (pad_left, pad_left)), mode='constant')
    elif pad_mode == 2:
        xp = np.pad(fmap_2d, ((0, 0), (pad_left, pad_left)), mode='reflect')
    else:
        raise ValueError(f"Unknown pad_mode {pad_mode}")
    _, T = xp.shape
    nout = T - kw + 1
    K = cin * kw
    out = np.zeros((K, nout), dtype=np.int8)
    for n in range(nout):
        k = 0
        for c in range(cin):
            for w in range(kw):
                out[k, n] = xp[c, n + w]
                k += 1
    return out


def im2col_2d_ref(fmap_3d, kh, kw):
    """fmap_3d: (Cin, Hin, Win) int8 (pre-padded). Returns (Cin*kh*kw, Hout*Wout) int8."""
    cin, hin, win = fmap_3d.shape
    hout = hin - kh + 1
    wout = win - kw + 1
    K = cin * kh * kw
    nout = hout * wout
    out = np.zeros((K, nout), dtype=np.int8)
    col = 0
    for h in range(hout):
        for w in range(wout):
            k = 0
            for c in range(cin):
                for khi in range(kh):
                    for kwi in range(kw):
                        out[k, col] = fmap_3d[c, h + khi, w + kwi]
                        k += 1
            col += 1
    return out


def run_im2col_hw(sim, fmap_flat, cin, hin, win, kh, kw, pad_left, pad_mode, nout):
    with tempfile.TemporaryDirectory() as tmp:
        fmap_f = os.path.join(tmp, 'fmap.hex')
        out_f = os.path.join(tmp, 'out.hex')
        iohex.write_int8_hex(fmap_f, fmap_flat)
        pa = [f'+cin={cin}', f'+hin={hin}', f'+win={win}',
              f'+kh={kh}', f'+kw={kw}',
              f'+padl={pad_left}', f'+padm={pad_mode}',
              f'+nout={nout}',
              f'+fmap={fmap_f}', f'+out={out_f}']
        r = subprocess.run(['vvp', sim] + pa, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr)
            raise RuntimeError("vvp failed")
        expected = cin * kh * kw * nout
        return iohex.read_int8_hex(out_f, count=expected)


def test_1d(sim, cin, win, kw, pad_left, pad_mode, label):
    fmap = np.random.randint(-128, 128, (cin, win), dtype=np.int8)
    golden = im2col_1d_ref(fmap, kw, pad_left, pad_mode)
    K, nout = golden.shape
    golden_flat = golden.flatten(order='F')
    hw_flat = run_im2col_hw(sim, fmap.flatten(), cin, 1, win, 1, kw,
                            pad_left, pad_mode, nout)
    ok = np.array_equal(golden_flat, hw_flat)
    tag = 'PASS' if ok else 'FAIL'
    print(f'{tag} {label}: cin={cin} win={win} kw={kw} padl={pad_left} '
          f'padm={pad_mode} -> K={K} N={nout}')
    if not ok:
        diffs = np.where(golden_flat != hw_flat)[0]
        for idx in diffs[:5]:
            print(f'  [{idx}] expected {int(golden_flat[idx])}, '
                  f'got {int(hw_flat[idx])}')
    return ok


def test_2d(sim, cin, hin, win, kh, kw, label):
    fmap = np.random.randint(-128, 128, (cin, hin, win), dtype=np.int8)
    golden = im2col_2d_ref(fmap, kh, kw)
    K, nout = golden.shape
    golden_flat = golden.flatten(order='F')
    hw_flat = run_im2col_hw(sim, fmap.flatten(), cin, hin, win, kh, kw,
                            0, 0, nout)
    ok = np.array_equal(golden_flat, hw_flat)
    tag = 'PASS' if ok else 'FAIL'
    print(f'{tag} {label}: cin={cin} hin={hin} win={win} kh={kh} kw={kw} '
          f'-> K={K} N={nout}')
    if not ok:
        diffs = np.where(golden_flat != hw_flat)[0]
        for idx in diffs[:5]:
            print(f'  [{idx}] expected {int(golden_flat[idx])}, '
                  f'got {int(hw_flat[idx])}')
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='all',
                        help='Configs to test: D,B,C,A or all')
    args = parser.parse_args()

    np.random.seed(42)
    sim = compile_sim()
    ok = True
    tid = 0
    cfgs = (args.config.upper().split(',')
            if args.config != 'all' else ['D', 'B', 'C', 'A'])

    if 'D' in cfgs:
        print('--- Config D: no padding ---')
        for cin, win, kw in [(64, 66, 3), (64, 66, 3), (64, 66, 3)]:
            tid += 1
            ok &= test_1d(sim, cin, win, kw, 0, 0, f'T{tid:02d}')
        for cin, win, kw in [(1, 5, 3), (2, 10, 3), (4, 8, 5), (1, 3, 1),
                             (1, 1, 1), (8, 20, 7)]:
            tid += 1
            ok &= test_1d(sim, cin, win, kw, 0, 0, f'T{tid:02d}')
        for _ in range(10):
            c = int(np.random.choice([1, 2, 4, 8, 16, 64]))
            w = int(np.random.choice([3, 5, 8, 16, 32, 64, 66, 128]))
            k = int(np.random.choice([1, 3, 5]))
            if w < k: w = k
            tid += 1
            ok &= test_1d(sim, c, w, k, 0, 0, f'T{tid:02d}')

    if 'B' in cfgs:
        print('--- Config B: zero padding ---')
        for cin, win, kw, pl in [(64, 128, 5, 2), (64, 128, 5, 2)]:
            tid += 1
            ok &= test_1d(sim, cin, win, kw, pl, 1, f'T{tid:02d}')
        for cin, win, kw, pl in [(1, 5, 3, 1), (2, 8, 5, 2), (4, 10, 3, 1),
                                 (1, 3, 3, 1), (1, 1, 3, 1)]:
            tid += 1
            ok &= test_1d(sim, cin, win, kw, pl, 1, f'T{tid:02d}')
        for _ in range(10):
            c = int(np.random.choice([1, 2, 4, 8, 16, 64]))
            w = int(np.random.choice([5, 8, 16, 32, 64, 128]))
            k = int(np.random.choice([3, 5, 7]))
            pl = (k - 1) // 2
            tid += 1
            ok &= test_1d(sim, c, w, k, pl, 1, f'T{tid:02d}')

    if 'C' in cfgs:
        print('--- Config C: reflect padding ---')
        for cin, win, kw, pl in [(64, 64, 3, 2), (64, 64, 3, 2)]:
            tid += 1
            ok &= test_1d(sim, cin, win, kw, pl, 2, f'T{tid:02d}')
        for cin, win, kw, pl in [(1, 4, 3, 2), (2, 8, 3, 2), (4, 6, 3, 2),
                                 (1, 3, 3, 2)]:
            tid += 1
            ok &= test_1d(sim, cin, win, kw, pl, 2, f'T{tid:02d}')
        for _ in range(10):
            c = int(np.random.choice([1, 2, 4, 8, 16, 64]))
            w = int(np.random.choice([4, 8, 16, 32, 64]))
            tid += 1
            ok &= test_1d(sim, c, w, 3, 2, 2, f'T{tid:02d}')

    if 'A' in cfgs:
        print('--- Config A: 2D im2col ---')
        for cin, hin, win, kh, kw in [(1, 2, 134, 2, 7), (1, 2, 134, 2, 7)]:
            tid += 1
            ok &= test_2d(sim, cin, hin, win, kh, kw, f'T{tid:02d}')
        for cin, hin, win, kh, kw in [(1, 2, 10, 2, 3), (1, 3, 8, 2, 3),
                                      (2, 2, 10, 2, 5), (1, 1, 7, 1, 3)]:
            tid += 1
            ok &= test_2d(sim, cin, hin, win, kh, kw, f'T{tid:02d}')
        for _ in range(10):
            c = int(np.random.choice([1, 2, 4]))
            h = int(np.random.choice([1, 2, 3, 4]))
            w = int(np.random.choice([5, 8, 16, 32]))
            kh = int(np.random.choice([1, 2, 3]))
            kw = int(np.random.choice([1, 3, 5, 7]))
            if h < kh: h = kh
            if w < kw: w = kw
            tid += 1
            ok &= test_2d(sim, c, h, w, kh, kw, f'T{tid:02d}')

    print()
    if ok:
        print(f'ALL IM2COL TESTS PASSED ({tid} tests)')
        sys.exit(0)
    else:
        print('SOME TESTS FAILED')
        sys.exit(1)


if __name__ == '__main__':
    main()
