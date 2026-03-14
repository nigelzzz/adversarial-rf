#!/usr/bin/env python
"""
Test LP-20 (lowpass cutoff=20) defense: AMC accuracy + CRC pass rate (with FEC).
Attacks: CW, EADL1. SNR: 0, 18. All modulations.
"""
import numpy as np
import torch
import pickle
from util.synth_txrx import (
    generate_burst, demodulate_burst, get_bits_per_symbol,
    FSK_MODS,
)
from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial, compute_accuracy
from util.defense import fft_topk_denoise


def lowpass_filter(x, cutoff_bin=20):
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    mask = torch.zeros_like(X)
    mask[:, :, :cutoff_bin] = 1.0
    X_filtered = X * mask
    return torch.fft.irfft(X_filtered, n=T, dim=2)


IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


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
    digital_mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4', 'CPFSK', 'GFSK']
    n_samples = 200
    n_bursts = 200
    attacks_to_test = ['cw', 'eadl1']
    defenses = {
        'None': lambda x: x,
        'Top-10': lambda x: fft_topk_denoise(x, topk=10),
        'Top-20': lambda x: fft_topk_denoise(x, topk=20),
        'LP-20': lambda x: lowpass_filter(x, cutoff_bin=20),
    }

    for attack_name in attacks_to_test:
        attack = create_attack(attack_name, wrapped_model, cfg)

        for snr in [0, 18]:
            print(f"\n{'='*130}")
            print(f"  {attack_name.upper()} Attack | SNR={snr} dB")
            print(f"{'='*130}")
            print(f"{'Mod':<8}{'Clean':>7}{'Att':>7}", end="")
            for dn in defenses:
                print(f"  {dn+' Acc':>10}", end="")
            # CRC columns for digital mods
            for dn in defenses:
                print(f"  {dn+' CRC':>10}", end="")
            print()
            print("-" * 130)

            for mod in mods:
                key = (mod.encode(), snr)
                if key not in rml_data:
                    continue
                samples = rml_data[key][:n_samples]
                x = torch.from_numpy(samples).float().to(device)
                true_idx = MOD_TO_IDX[mod]
                y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

                # Clean & attack accuracy
                clean_acc = compute_accuracy(model, x, y)
                x_adv = generate_adversarial(
                    attack, x, y,
                    wrapped_model=wrapped_model, ta_box='minmax',
                )
                att_acc = compute_accuracy(model, x_adv, y)

                # Defense accuracies
                def_accs = {}
                def_tensors = {}
                for dn, dfn in defenses.items():
                    x_def = dfn(x_adv)
                    def_accs[dn] = compute_accuracy(model, x_def, y)
                    def_tensors[dn] = x_def

                # CRC pass rate with FEC (only for digital mods)
                can_demod = mod in digital_mods
                bps = get_bits_per_symbol(mod) if can_demod else 0
                n_data_symbols = 16 - 2
                can_fec = can_demod and (mod not in FSK_MODS) and \
                          ((n_data_symbols * bps) // 2 - 6 >= 8)

                delta = (x_adv - x).cpu().numpy()
                rng = np.random.default_rng(42)

                crc_rates = {}
                if can_demod:
                    # Generate synthetic bursts with FEC
                    bursts = []
                    for _ in range(n_bursts):
                        b = generate_burst(
                            mod, n_symbols=16, n_pilots=2, sps=8,
                            beta=0.35, snr_db=snr, target_rms=0.006,
                            cfo_std=0.0, rng=rng, fec=can_fec,
                        )
                        bursts.append(b)

                    synth_iq = np.concatenate([b['iq_tensor'] for b in bursts], axis=0)
                    sample_idx = rng.integers(0, len(delta), size=n_bursts)
                    adv_synth = synth_iq + delta[sample_idx]
                    adv_synth_t = torch.from_numpy(adv_synth).float()

                    for dn, dfn in defenses.items():
                        defended_t = dfn(adv_synth_t.to(device))
                        defended_np = defended_t.cpu().numpy()

                        crc_pass = 0
                        for i in range(n_bursts):
                            def_iq = defended_np[i, 0, :] + 1j * defended_np[i, 1, :]
                            try:
                                iq_full = bursts[i].get('iq_full')
                                ws = bursts[i].get('iq_win_start')
                                ng = bursts[i].get('n_guard')
                                if iq_full is not None:
                                    iq_full = iq_full.copy()
                                    iq_full[ws:ws+len(def_iq)] = def_iq

                                result = demodulate_burst(
                                    def_iq, mod,
                                    n_pilots=bursts[i]['n_pilots'],
                                    pilot_symbols=bursts[i].get('pilot_symbols'),
                                    pilot_bits=bursts[i].get('pilot_bits'),
                                    sps=8, beta=0.35,
                                    pilot_positions=bursts[i].get('pilot_positions'),
                                    iq_full=iq_full, iq_win_start=ws, n_guard=ng,
                                    fec=can_fec, soft_demod=True,
                                    fec_coded_len=bursts[i].get('fec_coded_len'),
                                )
                                crc_pass += int(result['crc_pass'])
                            except Exception:
                                pass
                        crc_rates[dn] = crc_pass / n_bursts

                # Print row
                row = f"{mod:<8}{clean_acc*100:>6.1f}%{att_acc*100:>6.1f}%"
                for dn in defenses:
                    row += f"{def_accs[dn]*100:>11.1f}%"
                for dn in defenses:
                    if can_demod:
                        fec_tag = "(FEC)" if can_fec else "(noFEC)"
                        row += f"{crc_rates[dn]*100:>8.1f}%{fec_tag}"
                    else:
                        row += f"{'N/A':>12}"
                print(row)

            print()


if __name__ == '__main__':
    main()
