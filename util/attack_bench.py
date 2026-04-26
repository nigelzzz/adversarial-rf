"""
Phase 7: 5-attack x 2-device adversarial generation latency benchmark.

This module exposes :func:`run_attack_bench_5x2` which measures per-sample
adversarial-generation latency for the five paper attacks
(``fgsm``, ``pgd``, ``cw``, ``eadl1``, ``eaden``) on both CPU and GPU in a
single invocation, and writes a 10-row CSV plus a companion environment
metadata JSON under ``cfg.result_dir``.

Hyperparameter set actually used (relative to v1.0 paper defaults / D-05)
------------------------------------------------------------------------
The bench reuses :func:`util.sigguard_eval.create_attack` verbatim — it does
NOT duplicate the attack factory. The paper-default hyperparameters are
expected to be stamped into ``cfg`` by the caller (Plan 02 owns the CLI
defaults). When the bench function runs it sets:

  * ``cfg.attack_eps``    -> ``0.03``  (FGSM/PGD eps; D-05)
  * ``cfg.cw_c``          -> ``1.0``   (D-05)
  * ``cfg.cw_steps``      -> ``100``   (D-05)
  * ``cfg.cw_lr``         -> ``0.01``  (D-05)
  * ``cfg.ead_max_iterations`` -> ``100`` (D-05)
  * ``cfg.ta_box``        -> ``'unit'`` (D-05; default)

PGD alpha note: ``create_attack`` derives ``alpha = eps / 4`` so with
``eps=0.03`` we get ``alpha = 0.0075`` rather than the paper's quoted
``alpha=0.01``. This divergence is intentional: duplicating the attack
factory just to override one hyperparameter would defeat the purpose of
reusing the pipeline. The latency impact is negligible because PGD timing
is dominated by the inner forward/backward passes, not the step size.

Stratification choice
---------------------
* If ``snrs_test`` and ``test_idx`` are provided, samples are drawn via
  round-robin across (snr, label) buckets — equivalent to
  :func:`util.multi_attack_eval.build_snr_mod_index` cells.
* Otherwise the helper falls back to round-robin across label buckets only.
* A fixed seed (``2022``) keeps the stratified sample reproducible across
  invocations.

Outputs (under ``cfg.result_dir``)
----------------------------------
* ``attack_bench.csv``        — header comment + 10 data rows
* ``attack_bench_env.json``   — torch / cuda / cpu metadata + raw rows

Halts early with :class:`RuntimeError` if CUDA is unavailable on the host
(D-16): the project explicitly targets a single-GPU host and a CPU-only
report would be misleading.
"""

import json
import os
import platform
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial


# Paper-fixed attack list (D-05). Order matches CSV row order per device.
ATTACKS_5 = ['fgsm', 'pgd', 'cw', 'eadl1', 'eaden']

DEFAULT_N_SAMPLES = 512
DEFAULT_N_REPS = 5
DEFAULT_BATCH_SIZE = 128
DEFAULT_WARMUP = 3


def _collect_env() -> dict:
    """Snapshot host environment metadata for reproducibility (D-12)."""
    return {
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        'cpu_count': os.cpu_count(),
        'cpu_processor': platform.processor(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
    }


def _stratified_indices(
    lab_test: torch.Tensor,
    snrs_test: Optional[List[int]],
    n_samples: int,
    seed: int = 2022,
) -> torch.Tensor:
    """
    Round-robin draw of ``n_samples`` indices from ``lab_test`` so the
    benchmark sample is balanced across modulation (and SNR if provided).

    Parameters
    ----------
    lab_test : torch.Tensor
        Test split label tensor (1-D LongTensor on CPU).
    snrs_test : Optional[List[int]]
        Per-test-sample SNRs aligned with ``lab_test``. If ``None``, the
        sample is stratified across labels only.
    n_samples : int
        Total number of indices to draw.
    seed : int
        Shuffle seed for deterministic sampling.

    Returns
    -------
    torch.Tensor
        1-D LongTensor of length exactly ``n_samples``.
    """
    rng = np.random.default_rng(seed)
    labels = lab_test.cpu().numpy().astype(np.int64)
    n_total = labels.shape[0]
    if n_samples > n_total:
        raise ValueError(
            f"n_samples={n_samples} exceeds available test samples ({n_total})"
        )

    # Build per-bucket index lists, shuffled.
    buckets: Dict[Tuple, np.ndarray] = {}
    if snrs_test is None:
        for lbl in np.unique(labels):
            idx_list = np.where(labels == lbl)[0]
            rng.shuffle(idx_list)
            buckets[(int(lbl),)] = idx_list
    else:
        snrs = np.asarray(snrs_test, dtype=np.int64)
        if snrs.shape[0] != n_total:
            raise ValueError(
                f"snrs_test length {snrs.shape[0]} != lab_test length {n_total}"
            )
        for lbl in np.unique(labels):
            for snr in np.unique(snrs):
                mask = (labels == lbl) & (snrs == snr)
                idx_list = np.where(mask)[0]
                if idx_list.size == 0:
                    continue
                rng.shuffle(idx_list)
                buckets[(int(lbl), int(snr))] = idx_list

    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)

    # Round-robin draw.
    cursors = {k: 0 for k in bucket_keys}
    picked: List[int] = []
    while len(picked) < n_samples:
        progressed = False
        for k in bucket_keys:
            if len(picked) >= n_samples:
                break
            c = cursors[k]
            arr = buckets[k]
            if c < arr.size:
                picked.append(int(arr[c]))
                cursors[k] = c + 1
                progressed = True
        if not progressed:
            # Buckets exhausted before n_samples reached; fill from remaining.
            remaining = np.setdiff1d(
                np.arange(n_total, dtype=np.int64),
                np.asarray(picked, dtype=np.int64),
                assume_unique=False,
            )
            rng.shuffle(remaining)
            need = n_samples - len(picked)
            picked.extend(int(x) for x in remaining[:need])
            break

    return torch.as_tensor(picked[:n_samples], dtype=torch.long)


def _iter_batches(
    sig: torch.Tensor,
    lab: torch.Tensor,
    batch_size: int,
):
    """Yield ``(batch_x, batch_y)`` chunks of size up to ``batch_size``."""
    n = sig.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield sig[start:end], lab[start:end]


def _time_one_cell(
    attack,
    sig_iq: torch.Tensor,
    lab: torch.Tensor,
    wrapped_model,
    ta_box: str,
    batch_size: int,
    warmup: int,
    n_reps: int,
    device: str,
) -> Tuple[float, float, float]:
    """
    Time a single (attack, device) cell.

    Returns
    -------
    Tuple[float, float, float]
        ``(mean_ms_per_sample, std_ms_per_sample, total_seconds_across_reps)``.
        ``mean`` and ``std`` are computed over per-rep per-sample latencies
        with population standard deviation (``ddof=0``).
    """
    n_total_samples = sig_iq.shape[0]
    if n_total_samples == 0:
        return 0.0, 0.0, 0.0

    # Warmup: discard timings.
    for _ in range(max(0, warmup)):
        for batch_x, batch_y in _iter_batches(sig_iq, lab, batch_size):
            _ = generate_adversarial(
                attack,
                batch_x,
                batch_y,
                wrapped_model=wrapped_model,
                ta_box=ta_box,
            )

    per_sample_ms_list: List[float] = []
    total_seconds = 0.0
    for _ in range(n_reps):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch_x, batch_y in _iter_batches(sig_iq, lab, batch_size):
            _ = generate_adversarial(
                attack,
                batch_x,
                batch_y,
                wrapped_model=wrapped_model,
                ta_box=ta_box,
            )
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        block_seconds = t1 - t0
        total_seconds += block_seconds
        per_sample_ms_list.append((block_seconds / n_total_samples) * 1000.0)

    arr = np.asarray(per_sample_ms_list, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0)), float(total_seconds)


def _stamp_paper_defaults(cfg) -> None:
    """Force D-05 paper-default hyperparameters onto ``cfg`` in-place."""
    cfg.attack_eps = float(getattr(cfg, 'attack_eps', 0.03) or 0.03)
    if cfg.attack_eps != 0.03:
        cfg.attack_eps = 0.03
    cfg.cw_c = float(getattr(cfg, 'cw_c', 1.0) if getattr(cfg, 'cw_c', None) is not None else 1.0)
    if cfg.cw_c != 1.0:
        cfg.cw_c = 1.0
    cfg.cw_steps = int(getattr(cfg, 'cw_steps', 100) if getattr(cfg, 'cw_steps', None) else 100)
    if cfg.cw_steps != 100:
        cfg.cw_steps = 100
    cfg.cw_lr = float(
        getattr(cfg, 'cw_lr', 0.01) if getattr(cfg, 'cw_lr', None) is not None else 0.01
    )
    if cfg.cw_lr != 0.01:
        cfg.cw_lr = 0.01
    cfg.ead_max_iterations = int(
        getattr(cfg, 'ead_max_iterations', 100)
        if getattr(cfg, 'ead_max_iterations', None) else 100
    )
    if cfg.ead_max_iterations != 100:
        cfg.ead_max_iterations = 100
    if not hasattr(cfg, 'ta_box') or cfg.ta_box is None:
        cfg.ta_box = 'unit'


def run_attack_bench_5x2(
    model,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    cfg,
    logger,
    snrs_test: Optional[List[int]] = None,
    test_idx: Optional[np.ndarray] = None,
) -> None:
    """
    Run the Phase 7 5-attack x 2-device adversarial generation benchmark.

    Parameters
    ----------
    model : torch.nn.Module
        The base AWN classifier (returns ``(logits, regu_sum)`` from forward).
        It will be moved to CPU and CUDA in turn; its state_dict is reloaded
        between device passes (D-13 + Specifics).
    sig_test : torch.Tensor
        Test split IQ tensor of shape ``[N, 2, T]`` on host memory.
    lab_test : torch.Tensor
        Test split label tensor of shape ``[N]``.
    cfg : Any
        Config object. Must expose ``result_dir``. Honors the optional
        bench knobs ``bench_n_samples``, ``bench_n_reps``,
        ``bench_batch_size``, ``bench_warmup``, ``ta_box``.
    logger : logging.Logger
        Project logger (use :func:`util.logger.create_logger`).
    snrs_test : Optional[List[int]]
        Optional per-sample SNR list aligned with ``lab_test``; enables
        full ``(snr, mod)`` stratification.
    test_idx : Optional[np.ndarray]
        Optional original-dataset indices for the test split. Reserved for
        future stratification refinements; currently unused.

    Notes
    -----
    Halts immediately with :class:`RuntimeError` if CUDA is unavailable
    (D-16). Phase 7 explicitly compares CPU and GPU passes — a CPU-only
    report would be misleading.
    """
    # D-16: hard requirement.
    if not torch.cuda.is_available():
        raise RuntimeError("attack_bench requires CUDA. Halting (D-16).")

    # `test_idx` is accepted for forward compatibility with Plan 02 wiring.
    _ = test_idx  # silence linters; intentional.

    # D-05: stamp paper-default hyperparameters before the attack factory
    # consults cfg.
    _stamp_paper_defaults(cfg)

    n_samples = int(getattr(cfg, 'bench_n_samples', DEFAULT_N_SAMPLES))
    n_reps = int(getattr(cfg, 'bench_n_reps', DEFAULT_N_REPS))
    batch_size = int(getattr(cfg, 'bench_batch_size', DEFAULT_BATCH_SIZE))
    warmup = int(getattr(cfg, 'bench_warmup', DEFAULT_WARMUP))
    ta_box = str(getattr(cfg, 'ta_box', 'unit')).lower()

    logger.info(
        "Attack bench config: n_samples=%d n_reps=%d batch_size=%d warmup=%d ta_box=%s",
        n_samples, n_reps, batch_size, warmup, ta_box,
    )

    # Snapshot original state_dict on CPU for deterministic reload between
    # device passes (mitigates T-07-01 — in-place mutation by attacks).
    original_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }

    # Stratified sample (host memory) — built once.
    idx = _stratified_indices(lab_test.cpu(), snrs_test, n_samples)
    sig_strat = sig_test[idx].contiguous()
    lab_strat = lab_test[idx].contiguous()
    assert sig_strat.shape[0] == n_samples, (
        f"stratified sample size {sig_strat.shape[0]} != n_samples {n_samples}"
    )

    env = _collect_env()
    logger.info(
        "Env: torch=%s cuda=%s gpu=%s cpu=%s cores=%s",
        env['torch_version'], env['cuda_version'], env['gpu_name'],
        env['cpu_processor'], env['cpu_count'],
    )

    rows: List[dict] = []

    for device in ['cpu', 'cuda']:
        # Belt-and-suspenders: reload original weights on the target device
        # before each pass (D-13 + Specifics).
        reloaded = {k: v.to(device) for k, v in original_state.items()}
        model.load_state_dict(reloaded)
        model.to(device).eval()

        wrapped_model = Model01Wrapper(model).to(device).eval()
        sig_dev = sig_strat.to(device)
        lab_dev = lab_strat.to(device)

        for attack_name in tqdm(
            ATTACKS_5, desc=f"attack_bench[{device}]", leave=False,
        ):
            attack = create_attack(attack_name, wrapped_model, cfg)
            mean_ms, std_ms, total_s = _time_one_cell(
                attack=attack,
                sig_iq=sig_dev,
                lab=lab_dev,
                wrapped_model=wrapped_model,
                ta_box=ta_box,
                batch_size=batch_size,
                warmup=warmup,
                n_reps=n_reps,
                device=device,
            )
            rows.append({
                'attack': attack_name,
                'device': device,
                'batch_size': batch_size,
                'n_samples': n_samples,
                'n_reps': n_reps,
                'mean_ms_per_sample': mean_ms,
                'std_ms_per_sample': std_ms,
                'total_seconds': total_s,
            })
            logger.info(
                "[%s] %s: %.3f +/- %.3f ms/sample (total %.2fs)",
                device, attack_name, mean_ms, std_ms, total_s,
            )

    # Write CSV with `# env: ...` comment line and JSON sidecar (D-09, D-12).
    os.makedirs(cfg.result_dir, exist_ok=True)
    csv_path = os.path.join(cfg.result_dir, 'attack_bench.csv')
    json_path = os.path.join(cfg.result_dir, 'attack_bench_env.json')

    env_comment = (
        f"# env: torch={env['torch_version']} cuda={env['cuda_version']} "
        f"gpu={env['gpu_name']} cpu={env['cpu_processor']!r} "
        f"cores={env['cpu_count']}\n"
    )
    header = (
        "attack,device,batch_size,n_samples,n_reps,"
        "mean_ms_per_sample,std_ms_per_sample,total_seconds\n"
    )
    with open(csv_path, 'w') as f:
        f.write(env_comment)
        f.write(header)
        for r in rows:
            f.write(
                f"{r['attack']},{r['device']},{r['batch_size']},"
                f"{r['n_samples']},{r['n_reps']},"
                f"{r['mean_ms_per_sample']:.6f},"
                f"{r['std_ms_per_sample']:.6f},"
                f"{r['total_seconds']:.6f}\n"
            )

    with open(json_path, 'w') as f:
        json.dump(
            {
                'env': env,
                'rows': rows,
                'config': {
                    'n_samples': n_samples,
                    'n_reps': n_reps,
                    'batch_size': batch_size,
                    'warmup': warmup,
                    'ta_box': ta_box,
                    'attacks': ATTACKS_5,
                },
            },
            f,
            indent=2,
        )

    # Final summary table (5 rows x 2 device columns).
    by_attack: Dict[str, Dict[str, Tuple[float, float]]] = {a: {} for a in ATTACKS_5}
    for r in rows:
        by_attack[r['attack']][r['device']] = (
            r['mean_ms_per_sample'], r['std_ms_per_sample'],
        )
    logger.info(
        "Attack bench summary (ms/sample, mean +/- std, batch=%d, n=%d, reps=%d):",
        batch_size, n_samples, n_reps,
    )
    logger.info(
        "  %-8s | %-22s | %-22s",
        "attack", "cpu", "cuda",
    )
    for a in ATTACKS_5:
        cpu_stats = by_attack[a].get('cpu', (float('nan'), float('nan')))
        gpu_stats = by_attack[a].get('cuda', (float('nan'), float('nan')))
        logger.info(
            "  %-8s | %8.3f +/- %-8.3f | %8.3f +/- %-8.3f",
            a, cpu_stats[0], cpu_stats[1], gpu_stats[0], gpu_stats[1],
        )

    logger.info("Saved %s", csv_path)
    logger.info("Saved %s", json_path)
