"""Inference-time benchmark for AWN_quan_int8_simple.tflite (RF IQ classifier).

Input  : [1, 2, 128] int8  (I/Q signal, quantized)
Output : [1, 11]    int8  (modulation class logits, quantized)
"""

import argparse
import time

import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow import lite as tflite  # fallback for desktop


def load_iq_sample(pkl_path, mod=b'QPSK', snr=18):
    """Load one real RML2016.10a IQ sample. Returns float32 [2, 128] in [-1, 1]-ish range."""
    import pickle
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    sigs = data[(mod, snr)]            # [N, 2, 128]
    return sigs[0].astype(np.float32)  # [2, 128]


def quantize(x_f32, scale, zero_point):
    q = np.round(x_f32 / scale + zero_point)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(x_i8, scale, zero_point):
    return (x_i8.astype(np.float32) - zero_point) * scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file',
                        default='AWN_quan_int8_simple.tflite',
                        help='.tflite model to benchmark')
    parser.add_argument('--data_path',
                        default='./data/RML2016.10a_dict.pkl',
                        help='RML pickle for real IQ input (optional)')
    parser.add_argument('--use_random', action='store_true',
                        help='Use random IQ input instead of RML data')
    parser.add_argument('--num_runs', default=100, type=int,
                        help='Number of timed inference runs')
    parser.add_argument('--warmup', default=10, type=int,
                        help='Warmup runs (excluded from timing)')
    parser.add_argument('--num_threads', default=None, type=int)
    parser.add_argument('-e', '--ext_delegate',
                        help='external_delegate_library path (e.g. NPU)')
    parser.add_argument('-o', '--ext_delegate_options',
                        help='delegate options "k1:v1;k2:v2"')
    args = parser.parse_args()

    # ---- delegate ----------------------------------------------------------
    ext_delegate = None
    if args.ext_delegate is not None:
        ext_delegate_options = {}
        if args.ext_delegate_options is not None:
            for o in args.ext_delegate_options.split(';'):
                kv = o.split(':')
                if len(kv) == 2:
                    ext_delegate_options[kv[0].strip()] = kv[1].strip()
                else:
                    raise RuntimeError('Bad delegate option: ' + o)
        print(f'Loading external delegate {args.ext_delegate} '
              f'with {ext_delegate_options}')
        ext_delegate = [tflite.load_delegate(args.ext_delegate,
                                             ext_delegate_options)]

    # ---- interpreter -------------------------------------------------------
    interpreter = tflite.Interpreter(
        model_path=args.model_file,
        experimental_delegates=ext_delegate,
        num_threads=args.num_threads)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print('Input :', in_det['shape'], in_det['dtype'].__name__,
          'q=', in_det['quantization'])
    print('Output:', out_det['shape'], out_det['dtype'].__name__,
          'q=', out_det['quantization'])

    in_scale, in_zp = in_det['quantization']
    out_scale, out_zp = out_det['quantization']

    # ---- prepare input -----------------------------------------------------
    if args.use_random:
        iq = np.random.uniform(-0.02, 0.02, size=(2, 128)).astype(np.float32)
    else:
        try:
            iq = load_iq_sample(args.data_path)
            print(f'Loaded IQ sample from {args.data_path}, shape={iq.shape}')
        except Exception as e:
            print(f'Falling back to random input ({e})')
            iq = np.random.uniform(-0.02, 0.02,
                                   size=(2, 128)).astype(np.float32)

    if in_det['dtype'] == np.int8:
        x = quantize(iq, in_scale, in_zp)[None, :, :]   # [1,2,128] int8
    else:
        x = iq[None, :, :].astype(in_det['dtype'])

    interpreter.set_tensor(in_det['index'], x)

    # ---- warmup ------------------------------------------------------------
    for _ in range(args.warmup):
        interpreter.invoke()

    # ---- timed runs --------------------------------------------------------
    timings = np.empty(args.num_runs, dtype=np.float64)
    for i in range(args.num_runs):
        t0 = time.perf_counter()
        interpreter.invoke()
        timings[i] = (time.perf_counter() - t0) * 1000.0  # ms

    out = interpreter.get_tensor(out_det['index'])
    if out_det['dtype'] == np.int8:
        logits = dequantize(out[0], out_scale, out_zp)
    else:
        logits = out[0]
    pred = int(np.argmax(logits))

    # ---- report ------------------------------------------------------------
    print('\n=== Inference time over {} runs ==='.format(args.num_runs))
    print(f'  mean  : {timings.mean():8.3f} ms')
    print(f'  std   : {timings.std():8.3f} ms')
    print(f'  min   : {timings.min():8.3f} ms')
    print(f'  max   : {timings.max():8.3f} ms')
    print(f'  p50   : {np.percentile(timings, 50):8.3f} ms')
    print(f'  p95   : {np.percentile(timings, 95):8.3f} ms')
    print(f'  p99   : {np.percentile(timings, 99):8.3f} ms')
    print(f'  fps   : {1000.0 / timings.mean():8.2f}')
    print(f'\nPredicted class index: {pred}')
