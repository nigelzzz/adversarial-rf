#!/usr/bin/env python
"""
Visualize why IQ signals are spectrally peaky, and draw the spectral-gated defense architecture.

Generates:
1. PSD profiles of all 11 modulations (showing peaky vs flat)
2. Spectral flatness bar chart with formula annotation
3. Spectral-gated defense architecture flowchart
"""
import numpy as np
import torch
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    'font.size': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def main():
    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        rml_data = pickle.load(f, encoding='bytes')

    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4',
            'AM-DSB', 'GFSK', 'CPFSK', 'AM-SSB', 'WBFM']
    snr = 18
    n_samples = 200

    # ================================================================
    # Fig 1: PSD profiles for all modulations
    # ================================================================
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    flatness_vals = {}
    for i, mod in enumerate(mods):
        key = (mod.encode(), snr)
        samples = rml_data[key][:n_samples]
        x = torch.from_numpy(samples).float()

        # Compute average PSD (both I and Q channels)
        X = torch.fft.fft(x, n=128, dim=2)
        psd = (X.abs() ** 2).mean(dim=(0, 1))  # avg over samples and I/Q
        psd_db = 10 * torch.log10(psd + 1e-20)

        # Spectral flatness
        power = X.abs() ** 2 + 1e-20
        geo = torch.exp(torch.log(power).mean(dim=2))
        arith = power.mean(dim=2)
        sf = (geo / (arith + 1e-12)).mean(dim=1).mean().item()
        flatness_vals[mod] = sf

        # Significant bin count (10% of peak)
        mag = X.abs()
        peak = mag.max(dim=2, keepdim=True).values
        sig_bins = (mag > 0.1 * peak).float().sum(dim=2).max(dim=1).values.mean().item()

        ax = axes[i]
        freqs = np.arange(128)
        ax.plot(freqs, psd_db.numpy(), linewidth=1.2,
                color='#d32f2f' if sf > 0.4 else '#1565c0')
        ax.set_title(f'{mod}\nSF={sf:.4f}, SigBins={sig_bins:.0f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlim(0, 127)
        ax.set_ylabel('PSD (dB)' if i % 4 == 0 else '')
        ax.set_xlabel('FFT Bin' if i >= 8 else '')

        # Highlight: shade the "significant" region
        psd_np = psd.numpy()
        peak_val = psd_np.max()
        sig_mask = psd_np > 0.1 * peak_val
        ax.fill_between(freqs, psd_db.numpy().min(), psd_db.numpy(),
                         where=sig_mask, alpha=0.2,
                         color='#d32f2f' if sf > 0.4 else '#1565c0')

    # Hide last subplot
    axes[11].axis('off')

    fig.suptitle('PSD Profiles of All Modulations (SNR=18 dB)\n'
                 'Blue = Peaky (narrowband), Red = Flat (wideband)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('results/fig_psd_profiles.png', dpi=150, bbox_inches='tight')
    print("Saved: results/fig_psd_profiles.png")

    # ================================================================
    # Fig 2: Spectral Flatness bar chart with formula
    # ================================================================
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    colors = ['#d32f2f' if flatness_vals[m] > 0.4 else '#1565c0' for m in mods]
    bars = ax2.bar(mods, [flatness_vals[m] for m in mods], color=colors, edgecolor='black', linewidth=0.5)

    # Threshold line
    ax2.axhline(y=0.4, color='#f57c00', linewidth=2, linestyle='--', label='Threshold = 0.4')

    # Annotate values
    for bar, mod in zip(bars, mods):
        val = flatness_vals[mod]
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Formula box
    formula_text = (
        r'$\mathrm{Spectral\ Flatness} = \frac{\mathrm{Geometric\ Mean}(|X[k]|^2)}{\mathrm{Arithmetic\ Mean}(|X[k]|^2)}$'
        '\n\n'
        r'$= \frac{\exp\!\left(\frac{1}{N}\sum_{k=0}^{N-1} \ln |X[k]|^2\right)}{\frac{1}{N}\sum_{k=0}^{N-1} |X[k]|^2}$'
        '\n\n'
        'SF → 0: peaky (narrowband)\n'
        'SF → 1: flat (wideband / white noise)'
    )
    ax2.text(0.98, 0.95, formula_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    ax2.set_ylabel('Spectral Flatness (SF)')
    ax2.set_title('Spectral Flatness per Modulation (SNR=18 dB)\n'
                  'Only AM-SSB (SF=0.55) is wideband — all others are peaky (SF < 0.03)',
                  fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.set_ylim(0, 0.7)

    fig2.tight_layout()
    fig2.savefig('results/fig_spectral_flatness.png', dpi=150, bbox_inches='tight')
    print("Saved: results/fig_spectral_flatness.png")

    # ================================================================
    # Fig 3: Architecture diagram
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    ax3.set_xlim(0, 14)
    ax3.set_ylim(0, 8)
    ax3.axis('off')
    ax3.set_aspect('equal')

    def draw_box(ax, x, y, w, h, text, color='#e3f2fd', edgecolor='#1565c0',
                 fontsize=9, fontweight='normal', text_color='black'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                multialignment='center')

    def draw_arrow(ax, x1, y1, x2, y2, text='', color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, text, ha='center', va='bottom',
                    fontsize=8, color=color, fontstyle='italic')

    def draw_arrow_curved(ax, x1, y1, x2, y2, text='', color='black', rad=0.3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                    connectionstyle=f'arc3,rad={rad}'))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.3, my, text, ha='center', va='center',
                    fontsize=8, color=color, fontstyle='italic')

    # Title
    ax3.text(7, 7.6, 'Spectral-Gated Defense Architecture',
             ha='center', fontsize=16, fontweight='bold')
    ax3.text(7, 7.2, 'Cost: 1 FFT + 1 defense filter + 1 model inference',
             ha='center', fontsize=11, color='gray')

    # Input
    draw_box(ax3, 0.5, 5.5, 2.5, 1.0, 'Input IQ Signal\nx ∈ [N, 2, T]',
             color='#fff3e0', edgecolor='#e65100', fontsize=10, fontweight='bold')

    # FFT (shared)
    draw_box(ax3, 4.5, 5.5, 2.5, 1.0, 'FFT\nX = fft(x)\n(shared computation)',
             color='#f3e5f5', edgecolor='#7b1fa2', fontsize=9, fontweight='bold')

    # Arrow: Input → FFT
    draw_arrow(ax3, 3.0, 6.0, 4.5, 6.0)

    # Spectral Flatness computation
    draw_box(ax3, 4.5, 3.8, 2.5, 1.2,
             'Spectral Flatness\n'
             'SF = geo_mean(|X|²)\n'
             '     / arith_mean(|X|²)',
             color='#fce4ec', edgecolor='#c62828', fontsize=8, fontweight='bold')

    # Arrow: FFT → SF
    draw_arrow(ax3, 5.75, 5.5, 5.75, 5.0)

    # Decision diamond (as a rotated box)
    diamond_x, diamond_y = 5.75, 2.6
    diamond = plt.Polygon([
        [diamond_x, diamond_y + 0.6],
        [diamond_x + 1.0, diamond_y],
        [diamond_x, diamond_y - 0.6],
        [diamond_x - 1.0, diamond_y],
    ], closed=True, facecolor='#fff9c4', edgecolor='#f57f17', linewidth=2)
    ax3.add_patch(diamond)
    ax3.text(diamond_x, diamond_y, 'SF > 0.4 ?', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Arrow: SF → Decision
    draw_arrow(ax3, 5.75, 3.8, 5.75, 3.2)

    # YES branch → Quant32 (wideband)
    draw_box(ax3, 8.5, 2.0, 3.0, 1.2,
             'Quant32\n(per-sample quantization)\n32 discrete levels',
             color='#ffebee', edgecolor='#d32f2f', fontsize=9, fontweight='bold')
    draw_arrow(ax3, 6.75, 2.6, 8.5, 2.6, text='YES (wideband)', color='#d32f2f')

    # AM-SSB label
    ax3.text(10.0, 1.7, 'AM-SSB (SF≈0.55)',
             ha='center', fontsize=8, color='#d32f2f', fontstyle='italic')

    # NO branch → Top-K (narrowband)
    draw_box(ax3, 1.0, 2.0, 3.0, 1.2,
             'FFT Top-K\n(keep K largest bins)\nK=20 default',
             color='#e3f2fd', edgecolor='#1565c0', fontsize=9, fontweight='bold')
    draw_arrow(ax3, 4.75, 2.6, 4.0, 2.6, text='NO (narrowband)', color='#1565c0')

    # 10/11 mods label
    ax3.text(2.5, 1.7, '10/11 mods (SF < 0.03)',
             ha='center', fontsize=8, color='#1565c0', fontstyle='italic')

    # Both branches merge → Classifier
    draw_box(ax3, 4.5, 0.2, 2.5, 0.9, 'AWN Classifier\n(1 inference)',
             color='#e8f5e9', edgecolor='#2e7d36', fontsize=10, fontweight='bold')

    # Arrows to classifier
    draw_arrow_curved(ax3, 2.5, 2.0, 5.0, 1.1, color='#1565c0', rad=-0.3)
    draw_arrow_curved(ax3, 10.0, 2.0, 7.0, 1.1, color='#d32f2f', rad=0.3)

    # Reuse FFT arrow
    draw_arrow_curved(ax3, 7.0, 5.7, 8.5, 5.7, color='#7b1fa2', rad=0.0)
    draw_box(ax3, 8.5, 5.2, 3.0, 1.0,
             'Top-K Mask\n(reuse same FFT)\nX_filt = X · mask_topK',
             color='#ede7f6', edgecolor='#7b1fa2', fontsize=8)
    draw_arrow_curved(ax3, 10.0, 5.2, 2.5, 3.2, color='#7b1fa2', rad=0.4)
    ax3.text(8.0, 4.5, 'FFT shared!\nno recomputation',
             ha='center', fontsize=8, color='#7b1fa2', fontstyle='italic')

    fig3.savefig('results/fig_spectral_gated_arch.png', dpi=150, bbox_inches='tight')
    print("Saved: results/fig_spectral_gated_arch.png")

    # ================================================================
    # Print summary table
    # ================================================================
    print("\n" + "=" * 70)
    print("  Spectral Flatness Summary")
    print("=" * 70)
    print(f"{'Mod':<8} {'SF':>8} {'Category':>12} {'Route':>10}")
    print("-" * 42)
    for mod in mods:
        sf = flatness_vals[mod]
        cat = 'WIDEBAND' if sf > 0.4 else 'PEAKY'
        route = 'Quant32' if sf > 0.4 else 'Top-K'
        print(f"{mod:<8} {sf:>8.4f} {cat:>12} {route:>10}")
    print("-" * 42)
    print(f"Threshold = 0.4, Gap = {min(flatness_vals[m] for m in mods if flatness_vals[m] > 0.4):.4f} "
          f"vs {max(flatness_vals[m] for m in mods if flatness_vals[m] < 0.4):.4f} "
          f"(ratio = {min(flatness_vals[m] for m in mods if flatness_vals[m] > 0.4) / max(flatness_vals[m] for m in mods if flatness_vals[m] < 0.4):.0f}x)")


if __name__ == '__main__':
    main()
