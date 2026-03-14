#!/usr/bin/env python
"""
Defense Ensemble Voting: apply multiple defenses, classify each, majority vote.

No need to know modulation beforehand — the ensemble handles it.

Approaches tested:
1. Ensemble-3: Top-20 + LP-20 + Quant32 (diverse: spectral, lowpass, quantization)
2. Ensemble-5: Top-10 + Top-20 + LP-20 + Quant32 + Median5
3. Confidence-pick: apply all, pick highest softmax confidence
4. Vote2+LP: Multi-K vote (Top-10/20), if disagree → try LP-20 as tiebreaker
"""
import numpy as np
import torch
import torch.nn.functional as F_torch
import pickle
from collections import Counter

from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial, compute_accuracy
from util.defense import fft_topk_denoise


# ---- Defense functions ----
def lowpass_filter(x, cutoff_bin=20):
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    mask = torch.zeros_like(X)
    mask[:, :, :cutoff_bin] = 1.0
    return torch.fft.irfft(X * mask, n=T, dim=2)

def input_quantize(x, n_levels=32):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return x_quant * (x_max - x_min) + x_min

def median_filter(x, kernel_size=5):
    N, C, T = x.shape
    pad = kernel_size // 2
    x_padded = F_torch.pad(x, (pad, pad), mode='reflect')
    windows = x_padded.unfold(dimension=2, size=kernel_size, step=1)
    return windows.median(dim=3).values

def moving_avg_smooth(x, kernel_size=5):
    N, C, T = x.shape
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
    x_flat = x.view(N * C, 1, T)
    x_smooth = F_torch.conv1d(x_flat, kernel, padding=pad)
    return x_smooth.view(N, C, T)


IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


@torch.no_grad()
def classify_batch(model, x):
    """Returns (predictions, logits)."""
    logits, _ = model(x)
    return logits.argmax(dim=1), logits


@torch.no_grad()
def ensemble_vote(x_adv, model, defense_dict):
    """
    Apply each defense, classify, majority vote per sample.
    Returns: final predictions [N], selected defense name per sample.
    """
    model.eval()
    N = x_adv.shape[0]
    n_def = len(defense_dict)

    all_preds = []  # [n_def, N]
    all_confs = []  # [n_def, N]
    all_filtered = []
    def_names = list(defense_dict.keys())

    for name, dfn in defense_dict.items():
        x_def = dfn(x_adv)
        all_filtered.append(x_def)
        logits, _ = model(x_def)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confs = probs.max(dim=1).values
        all_preds.append(preds)
        all_confs.append(confs)

    preds_stack = torch.stack(all_preds, dim=0)  # [n_def, N]
    confs_stack = torch.stack(all_confs, dim=0)  # [n_def, N]

    # Majority vote per sample
    final_preds = torch.zeros(N, dtype=torch.long, device=x_adv.device)
    selected_def = []

    for i in range(N):
        sample_preds = preds_stack[:, i].cpu().tolist()
        counter = Counter(sample_preds)
        majority_pred, majority_count = counter.most_common(1)[0]
        final_preds[i] = majority_pred

        # Among defenses that voted for majority, pick highest confidence
        best_conf = -1
        best_def = def_names[0]
        for j, name in enumerate(def_names):
            if sample_preds[j] == majority_pred and confs_stack[j, i].item() > best_conf:
                best_conf = confs_stack[j, i].item()
                best_def = name
        selected_def.append(best_def)

    return final_preds, selected_def


@torch.no_grad()
def confidence_pick(x_adv, model, defense_dict):
    """Apply each defense, pick the one with highest confidence per sample."""
    model.eval()
    N = x_adv.shape[0]

    best_preds = torch.zeros(N, dtype=torch.long, device=x_adv.device)
    best_conf = torch.full((N,), -1.0, device=x_adv.device)
    selected_def = [''] * N
    def_names = list(defense_dict.keys())

    for name, dfn in defense_dict.items():
        x_def = dfn(x_adv)
        logits, _ = model(x_def)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        confs = probs.max(dim=1).values

        better = confs > best_conf
        best_preds[better] = preds[better]
        best_conf[better] = confs[better]
        for i in range(N):
            if better[i]:
                selected_def[i] = name

    return best_preds, selected_def


@torch.no_grad()
def vote2_lp_tiebreak(x_adv, model):
    """
    Multi-K Vote (Top-10 vs Top-20).
    If agree → use Top-10.
    If disagree → use LP-20 as tiebreaker.
    """
    model.eval()
    N = x_adv.shape[0]

    x_t10 = fft_topk_denoise(x_adv, topk=10)
    x_t20 = fft_topk_denoise(x_adv, topk=20)
    x_lp20 = lowpass_filter(x_adv, cutoff_bin=20)

    logits_t10, _ = model(x_t10)
    logits_t20, _ = model(x_t20)
    logits_lp20, _ = model(x_lp20)

    pred_t10 = logits_t10.argmax(1)
    pred_t20 = logits_t20.argmax(1)
    pred_lp20 = logits_lp20.argmax(1)

    agree = (pred_t10 == pred_t20)

    # Start with LP-20 as default (for disagreement cases)
    final_preds = pred_lp20.clone()
    selected = ['LP-20'] * N

    # Where Top-10 and Top-20 agree → use Top-10 prediction (aggressive filtering safe)
    final_preds[agree] = pred_t10[agree]
    for i in range(N):
        if agree[i]:
            selected[i] = 'Top-10'

    # Where disagree: majority vote among t10, t20, lp20
    disagree = ~agree
    if disagree.any():
        for i in range(N):
            if disagree[i]:
                votes = [pred_t10[i].item(), pred_t20[i].item(), pred_lp20[i].item()]
                counter = Counter(votes)
                majority = counter.most_common(1)[0][0]
                final_preds[i] = majority
                # Pick defense that voted for majority
                if pred_t20[i].item() == majority:
                    selected[i] = 'Top-20'
                elif pred_lp20[i].item() == majority:
                    selected[i] = 'LP-20'
                else:
                    selected[i] = 'Top-10'

    return final_preds, selected


def main():
    fix_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, model_name='awn')
    model.load_state_dict(torch.load('./checkpoint/2016.10a_AWN.pkl', map_location=device))
    model.eval()

    cfg.cw_c = 10.0
    cfg.cw_steps = 200
    cfg.cw_lr = 0.005
    cfg.attack_eps = 0.1
    cfg.ta_box = 'minmax'
    cfg.num_classes = 11

    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        rml_data = pickle.load(f, encoding='bytes')

    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4',
            'AM-DSB', 'GFSK', 'CPFSK', 'AM-SSB', 'WBFM']
    n_samples = 200
    attacks_to_test = ['cw', 'eadl1']

    # Defense ensembles
    ens3 = {
        'Top-20': lambda x: fft_topk_denoise(x, topk=20),
        'LP-20': lambda x: lowpass_filter(x, cutoff_bin=20),
        'Quant32': lambda x: input_quantize(x, n_levels=32),
    }
    ens5 = {
        'Top-10': lambda x: fft_topk_denoise(x, topk=10),
        'Top-20': lambda x: fft_topk_denoise(x, topk=20),
        'LP-20': lambda x: lowpass_filter(x, cutoff_bin=20),
        'Quant32': lambda x: input_quantize(x, n_levels=32),
        'Median5': lambda x: median_filter(x, kernel_size=5),
    }

    for attack_name in attacks_to_test:
        attack = create_attack(attack_name, wrapped_model, cfg)

        for snr in [0, 18]:
            print(f"\n{'='*120}")
            print(f"  {attack_name.upper()} | SNR={snr}")
            print(f"{'='*120}")
            print(f"{'Mod':<8}{'Clean':>7}{'Att':>7}{'Top-10':>8}{'Top-20':>8}"
                  f"{'LP-20':>8}{'Qnt32':>8}"
                  f"{'Ens3':>8}{'Ens5':>8}{'ConfPk':>8}{'V2+LP':>8}")
            print("-" * 120)

            # Accumulators for average
            all_accs = {k: [] for k in ['clean', 'att', 'top10', 'top20', 'lp20', 'qnt32',
                                         'ens3', 'ens5', 'confpk', 'v2lp']}

            for mod in mods:
                key = (mod.encode(), snr)
                if key not in rml_data:
                    continue
                samples = rml_data[key][:n_samples]
                x = torch.from_numpy(samples).float().to(device)
                true_idx = MOD_TO_IDX[mod]
                y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

                clean_acc = compute_accuracy(model, x, y)
                x_adv = generate_adversarial(
                    attack, x, y,
                    wrapped_model=wrapped_model, ta_box='minmax',
                )
                att_acc = compute_accuracy(model, x_adv, y)

                # Individual defenses
                t10_acc = compute_accuracy(model, fft_topk_denoise(x_adv, topk=10), y)
                t20_acc = compute_accuracy(model, fft_topk_denoise(x_adv, topk=20), y)
                lp20_acc = compute_accuracy(model, lowpass_filter(x_adv, cutoff_bin=20), y)
                q32_acc = compute_accuracy(model, input_quantize(x_adv, n_levels=32), y)

                # Ensemble-3
                ens3_preds, _ = ensemble_vote(x_adv, model, ens3)
                ens3_acc = (ens3_preds == y).float().mean().item()

                # Ensemble-5
                ens5_preds, _ = ensemble_vote(x_adv, model, ens5)
                ens5_acc = (ens5_preds == y).float().mean().item()

                # Confidence pick (from ens5 set)
                conf_preds, _ = confidence_pick(x_adv, model, ens5)
                conf_acc = (conf_preds == y).float().mean().item()

                # Vote2 + LP tiebreak
                v2lp_preds, _ = vote2_lp_tiebreak(x_adv, model)
                v2lp_acc = (v2lp_preds == y).float().mean().item()

                print(f"{mod:<8}{clean_acc*100:>6.1f}%{att_acc*100:>6.1f}%"
                      f"{t10_acc*100:>7.1f}%{t20_acc*100:>7.1f}%"
                      f"{lp20_acc*100:>7.1f}%{q32_acc*100:>7.1f}%"
                      f"{ens3_acc*100:>7.1f}%{ens5_acc*100:>7.1f}%"
                      f"{conf_acc*100:>7.1f}%{v2lp_acc*100:>7.1f}%")

                all_accs['clean'].append(clean_acc)
                all_accs['att'].append(att_acc)
                all_accs['top10'].append(t10_acc)
                all_accs['top20'].append(t20_acc)
                all_accs['lp20'].append(lp20_acc)
                all_accs['qnt32'].append(q32_acc)
                all_accs['ens3'].append(ens3_acc)
                all_accs['ens5'].append(ens5_acc)
                all_accs['confpk'].append(conf_acc)
                all_accs['v2lp'].append(v2lp_acc)

            # Average row
            print("-" * 120)
            avg = {k: np.mean(v) for k, v in all_accs.items()}
            print(f"{'AVG':<8}{avg['clean']*100:>6.1f}%{avg['att']*100:>6.1f}%"
                  f"{avg['top10']*100:>7.1f}%{avg['top20']*100:>7.1f}%"
                  f"{avg['lp20']*100:>7.1f}%{avg['qnt32']*100:>7.1f}%"
                  f"{avg['ens3']*100:>7.1f}%{avg['ens5']*100:>7.1f}%"
                  f"{avg['confpk']*100:>7.1f}%{avg['v2lp']*100:>7.1f}%")
            print()


if __name__ == '__main__':
    main()
