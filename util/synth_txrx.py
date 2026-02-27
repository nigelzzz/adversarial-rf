"""
Synthetic TX/RX chain for CRC experiment.

Pure numpy module (no ML dependencies). Provides:
- CRC-8 computation and verification
- Constellation mapping for BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4
- FSK modulation/demodulation for CPFSK and GFSK
- TX chain: bit generation, modulation, pulse shaping, channel noise
- RX chain: matched filter, timing recovery, phase recovery, demodulation, CRC check
"""

import numpy as np
from util.utils import rrc_filter

# ---------------------------------------------------------------------------
# Modulation type sets
# ---------------------------------------------------------------------------

FSK_MODS = {'CPFSK', 'GFSK'}
CONSTELLATION_MODS = {'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4'}
ALL_DIGITAL_MODS = FSK_MODS | CONSTELLATION_MODS

# ---------------------------------------------------------------------------
# CRC-8 (polynomial 0x07: x^8 + x^2 + x + 1)
# ---------------------------------------------------------------------------

CRC8_POLY = 0x07

def _crc8_table():
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table[i] = crc
    return table

_CRC8_TABLE = _crc8_table()


def crc8(data_bits):
    """Compute CRC-8 over a bit array. Returns 8-bit numpy array."""
    data_bits = np.asarray(data_bits, dtype=np.uint8)
    # Pack bits into bytes
    n_bits = len(data_bits)
    # Pad to multiple of 8
    pad = (8 - n_bits % 8) % 8
    padded = np.concatenate([data_bits, np.zeros(pad, dtype=np.uint8)])
    n_bytes = len(padded) // 8
    byte_arr = np.packbits(padded)[:n_bytes]

    crc = np.uint8(0)
    for b in byte_arr:
        crc = _CRC8_TABLE[int(crc ^ b)]

    # Return as 8-bit array
    return np.unpackbits(np.array([crc], dtype=np.uint8))


def crc8_check(frame_bits):
    """Verify CRC on data+crc frame. Returns True if CRC passes."""
    frame_bits = np.asarray(frame_bits, dtype=np.uint8)
    if len(frame_bits) < 9:
        return False
    data_bits = frame_bits[:-8]
    expected_crc = frame_bits[-8:]
    computed_crc = crc8(data_bits)
    return np.array_equal(computed_crc, expected_crc)


# ---------------------------------------------------------------------------
# Constellation definitions (unit-energy, Gray-coded)
# ---------------------------------------------------------------------------

def _gray_code(n_bits):
    """Generate Gray code sequence for n_bits."""
    n = 1 << n_bits
    return [i ^ (i >> 1) for i in range(n)]


def _qam_constellation(M_side):
    """Build square QAM constellation (M_side x M_side), Gray-coded, unit energy."""
    n_bits_per_dim = int(np.log2(M_side))
    gray = _gray_code(n_bits_per_dim)

    points = []
    bit_labels = []
    for yi, gy in enumerate(gray):
        for xi, gx in enumerate(gray):
            # Map grid indices to symmetric coordinates
            real = 2 * xi - (M_side - 1)
            imag = 2 * yi - (M_side - 1)
            points.append(complex(real, imag))
            # Bit label = concatenation of gray_y bits and gray_x bits
            label = (gy << n_bits_per_dim) | gx
            bit_labels.append(label)

    points = np.array(points)
    # Normalize to unit average energy
    rms = np.sqrt(np.mean(np.abs(points) ** 2))
    points /= rms

    # Build lookup: bit_labels[i] -> points[i]
    # Re-order so index = symbol integer value
    M = M_side * M_side
    ordered = np.zeros(M, dtype=complex)
    for i, lbl in enumerate(bit_labels):
        ordered[lbl] = points[i]

    return ordered


# Pre-computed constellations
_CONSTELLATIONS = {}


def _build_constellations():
    if _CONSTELLATIONS:
        return

    # BPSK: {0: -1, 1: +1}
    _CONSTELLATIONS['BPSK'] = np.array([-1.0 + 0j, 1.0 + 0j])

    # QPSK: Gray-coded, 2 bits per symbol
    # 00->+1+1j, 01->-1+1j, 11->-1-1j, 10->+1-1j (Gray: 0,1,3,2)
    qpsk = np.array([
        1 + 1j,   # 00
        -1 + 1j,  # 01
        1 - 1j,   # 10
        -1 - 1j,  # 11
    ]) / np.sqrt(2)
    _CONSTELLATIONS['QPSK'] = qpsk

    # 8PSK: Gray-coded phase points
    gray8 = _gray_code(3)  # [0,1,3,2,6,7,5,4]
    phases = np.array([2 * np.pi * g / 8 for g in range(8)])
    psk8 = np.zeros(8, dtype=complex)
    for i, g in enumerate(gray8):
        psk8[g] = np.exp(1j * phases[i])
    _CONSTELLATIONS['8PSK'] = psk8

    # QAM16: 4x4 grid
    _CONSTELLATIONS['QAM16'] = _qam_constellation(4)

    # QAM64: 8x8 grid
    _CONSTELLATIONS['QAM64'] = _qam_constellation(8)

    # PAM4: 4-level pulse amplitude modulation, Gray-coded, unit energy
    # Gray: index 0 (00) -> -3, 1 (01) -> -1, 2 (10) -> +3, 3 (11) -> +1
    pam4 = np.array([-3.0, -1.0, 3.0, 1.0]) / np.sqrt(5.0)
    _CONSTELLATIONS['PAM4'] = pam4.astype(complex)


def get_constellation(mod_type):
    """Get unit-energy constellation points for the given modulation."""
    _build_constellations()
    mod = mod_type.upper() if isinstance(mod_type, str) else mod_type.decode().upper()
    if mod not in _CONSTELLATIONS:
        raise ValueError(f"Unknown modulation: {mod_type}")
    return _CONSTELLATIONS[mod].copy()


def get_bits_per_symbol(mod_type):
    """Return bits per symbol for the modulation type."""
    mod = mod_type.upper() if isinstance(mod_type, str) else mod_type.decode().upper()
    return {
        'BPSK': 1, 'QPSK': 2, '8PSK': 3, 'QAM16': 4, 'QAM64': 6,
        'PAM4': 2, 'CPFSK': 1, 'GFSK': 1,
    }[mod]


def get_mod_order(mod_type):
    """Return modulation order M for phase recovery (constellation mods only)."""
    mod = mod_type.upper() if isinstance(mod_type, str) else mod_type.decode().upper()
    return {
        'BPSK': 2, 'QPSK': 4, '8PSK': 8, 'QAM16': 4, 'QAM64': 4,
        'PAM4': 4,
    }.get(mod, 2)


# ---------------------------------------------------------------------------
# Bit <-> Symbol mapping (constellation mods)
# ---------------------------------------------------------------------------

def bits_to_symbols(bits, mod_type):
    """Map bit array to constellation symbols."""
    bits = np.asarray(bits, dtype=np.uint8)
    bps = get_bits_per_symbol(mod_type)
    constellation = get_constellation(mod_type)

    if len(bits) % bps != 0:
        raise ValueError(f"Bit length {len(bits)} not divisible by {bps}")

    n_sym = len(bits) // bps
    symbols = np.zeros(n_sym, dtype=complex)
    for i in range(n_sym):
        bit_group = bits[i * bps:(i + 1) * bps]
        idx = 0
        for b in bit_group:
            idx = (idx << 1) | int(b)
        symbols[i] = constellation[idx]

    return symbols


def symbols_to_bits(symbols, mod_type):
    """Hard-decision demap: nearest constellation point -> bits."""
    constellation = get_constellation(mod_type)
    bps = get_bits_per_symbol(mod_type)
    M = len(constellation)

    symbols = np.asarray(symbols)
    n_sym = len(symbols)
    bits = np.zeros(n_sym * bps, dtype=np.uint8)

    for i, sym in enumerate(symbols):
        # Find nearest constellation point
        dists = np.abs(constellation - sym)
        idx = np.argmin(dists)
        # Convert index to bits
        for j in range(bps):
            bits[i * bps + (bps - 1 - j)] = (idx >> j) & 1

    return bits


# ---------------------------------------------------------------------------
# FSK helpers
# ---------------------------------------------------------------------------

def _gaussian_freq_filter(bt, sps, span=4):
    """Gaussian frequency pulse filter for GFSK.

    Args:
        bt: Bandwidth-time product (e.g. 0.5)
        sps: Samples per symbol
        span: Filter span in symbols (+/- span)

    Returns:
        Normalized Gaussian filter (DC gain = 1)
    """
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)  # in symbol periods
    sigma_s = sigma * sps  # in samples
    n = np.arange(-span * sps, span * sps + 1)
    g = np.exp(-n**2 / (2 * sigma_s**2))
    g /= g.sum()
    return g


# ---------------------------------------------------------------------------
# TX chain
# ---------------------------------------------------------------------------

def _normalize_mod(mod_type):
    """Normalize modulation type string to uppercase."""
    return mod_type.upper() if isinstance(mod_type, str) else mod_type.decode().upper()


def generate_burst(mod_type, n_symbols=16, n_pilots=2, sps=8, beta=0.35,
                   snr_db=18.0, target_rms=0.006, cfo_std=0.015,
                   rng=None, n_guard=16):
    """
    Generate a single burst with known bits, CRC, and pilot symbols/bits.

    Supports constellation mods (BPSK, QPSK, 8PSK, QAM16, QAM64, PAM4)
    and FSK mods (CPFSK, GFSK).

    Args:
        n_guard: Number of guard symbols on each side (constellation mods).
                 Guard symbols provide filter context so the RX matched
                 filter can operate on the full signal without edge ISI.

    Returns:
        dict with keys:
            iq_tensor: [1, 2, n_symbols*sps] numpy array
            iq_complex: complex baseband samples (128-sample window)
            iq_full: full signal with guards (for matched-filter demod)
            data_bits: original data bits (before CRC)
            frame_bits: data_bits + CRC
            pilot_symbols: known pilot constellation points (None for FSK)
            pilot_bits: known pilot bit values (None for constellation)
            pilot_positions: symbol indices of pilots
            n_pilots: number of pilot symbols/bits
            mod_type: modulation type string
            snr_db: SNR used
            cfo, phase0: channel impairments
    """
    if rng is None:
        rng = np.random.default_rng()

    mod = _normalize_mod(mod_type)

    if mod in FSK_MODS:
        return _generate_burst_fsk(mod, n_symbols, n_pilots, sps,
                                   snr_db, target_rms, cfo_std, rng)
    elif mod in CONSTELLATION_MODS:
        return _generate_burst_constellation(mod, n_symbols, n_pilots, sps,
                                             beta, snr_db, target_rms,
                                             cfo_std, rng, n_guard=n_guard)
    else:
        raise ValueError(f"Unknown modulation: {mod_type}. "
                         f"Supported: {sorted(ALL_DIGITAL_MODS)}")


def _generate_burst_constellation(mod, n_symbols, n_pilots, sps, beta,
                                  snr_db, target_rms, cfo_std, rng,
                                  n_guard=16):
    """Generate burst for constellation-based modulations.

    Uses guard symbols before/after the data+pilot region so the TX RRC
    filter transients settle before the extraction window.  The full
    signal (with guards) is stored for RX use — applying the RX matched
    filter to the full signal avoids edge ISI that would otherwise
    corrupt QAM64 symbols on a short 128-sample window.

    Returns both:
      - iq_tensor / iq_complex : 128-sample window (for AWN classifier)
      - iq_full : full signal with guards (for matched-filter demod)
    """
    bps = get_bits_per_symbol(mod)
    constellation = get_constellation(mod)
    n_data_symbols = n_symbols - n_pilots

    # Generate random data bits, append CRC
    n_data_bits = n_data_symbols * bps - 8  # Reserve 8 bits for CRC
    if n_data_bits < 1:
        raise ValueError(
            f"Not enough data symbols for CRC. Need at least "
            f"ceil(9/{bps})+{n_pilots} symbols, got {n_symbols}")
    data_bits = rng.integers(0, 2, size=n_data_bits).astype(np.uint8)
    crc_bits = crc8(data_bits)
    frame_bits = np.concatenate([data_bits, crc_bits])

    # Map frame bits to data symbols
    data_symbols = bits_to_symbols(frame_bits, mod)

    # Generate known pilot symbols
    pilot_indices = rng.integers(0, len(constellation), size=n_pilots)
    pilot_symbols = constellation[pilot_indices]

    # Place pilots at start and end for better CFO estimation baseline.
    n_start = (n_pilots + 1) // 2
    n_end = n_pilots - n_start
    start_pilots = pilot_symbols[:n_start]
    end_pilots = pilot_symbols[n_start:]
    all_symbols = np.concatenate([start_pilots, data_symbols, end_pilots])

    pilot_positions = list(range(n_start)) + list(
        range(n_symbols - n_end, n_symbols)
    )

    # Guard symbols (random) on each side for RRC filter warm-up
    guard_before = constellation[
        rng.integers(0, len(constellation), size=n_guard)]
    guard_after = constellation[
        rng.integers(0, len(constellation), size=n_guard)]
    full_symbols = np.concatenate([guard_before, all_symbols, guard_after])

    # Upsample and pulse shape the FULL sequence (guards included)
    n_full = len(full_symbols)
    n_full_samples = n_full * sps
    upsampled = np.zeros(n_full_samples, dtype=complex)
    upsampled[::sps] = full_symbols

    # RRC filter — use 16-symbol span for reduced truncation ISI
    num_taps = min(16 * sps + 1, n_full_samples // 2)
    if num_taps % 2 == 0:
        num_taps += 1
    h = rrc_filter(beta, sps, num_taps=num_taps)
    shaped_full = np.convolve(upsampled, h, mode='same')

    # Scale the FULL signal to target RMS
    n_samples = n_symbols * sps
    win_start = n_guard * sps
    shaped_window = shaped_full[win_start:win_start + n_samples]
    current_rms = np.sqrt(np.mean(np.abs(shaped_window) ** 2))
    rms_scale = target_rms * (1.0 + 0.05 * rng.standard_normal())
    rms_scale = max(rms_scale, target_rms * 0.5)
    if current_rms > 0:
        scale_factor = rms_scale / current_rms
        shaped_full *= scale_factor

    # Apply carrier frequency offset and random phase to FULL signal
    cfo = cfo_std * rng.standard_normal()
    phase0 = rng.uniform(0, 2 * np.pi)
    n_idx = np.arange(n_full_samples)
    phase_ramp = np.exp(1j * (2 * np.pi * cfo * n_idx + phase0))
    shaped_full = shaped_full * phase_ramp

    # Add AWGN to FULL signal
    signal_power = np.mean(np.abs(shaped_full) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(n_full_samples)
        + 1j * rng.standard_normal(n_full_samples)
    )
    noisy_full = shaped_full + noise

    # Extract the middle 128-sample window (for AWN classifier / iq_tensor)
    noisy_window = noisy_full[win_start:win_start + n_samples]

    # Build IQ tensor [1, 2, T]
    iq_tensor = np.zeros((1, 2, n_samples), dtype=np.float32)
    iq_tensor[0, 0, :] = noisy_window.real.astype(np.float32)
    iq_tensor[0, 1, :] = noisy_window.imag.astype(np.float32)

    return {
        'iq_tensor': iq_tensor,
        'iq_complex': noisy_window,
        'iq_full': noisy_full,
        'iq_win_start': win_start,
        'n_guard': n_guard,
        'data_bits': data_bits,
        'frame_bits': frame_bits,
        'pilot_symbols': pilot_symbols,
        'pilot_bits': None,
        'pilot_positions': pilot_positions,
        'n_pilots': n_pilots,
        'mod_type': mod,
        'snr_db': snr_db,
        'symbols_tx': all_symbols,
        'cfo': cfo,
        'phase0': phase0,
    }


def _generate_burst_fsk(mod, n_symbols, n_pilots, sps,
                        snr_db, target_rms, cfo_std, rng,
                        h=0.5, bt=None):
    """Generate burst for FSK modulations (CPFSK, GFSK).

    CPFSK: rectangular frequency pulse, mod index h=0.5 (MSK)
    GFSK: Gaussian-filtered frequency pulse, BT=0.5
    """
    if mod == 'GFSK' and bt is None:
        bt = 0.5

    bps = 1  # binary FSK
    n_data_symbols = n_symbols - n_pilots
    n_data_bits = n_data_symbols * bps - 8  # Reserve 8 for CRC

    if n_data_bits < 1:
        raise ValueError(
            f"Not enough data symbols for CRC. Need at least "
            f"{9 + n_pilots} symbols, got {n_symbols}")

    data_bits = rng.integers(0, 2, size=n_data_bits).astype(np.uint8)
    crc_bits = crc8(data_bits)
    frame_bits = np.concatenate([data_bits, crc_bits])

    # Pilot bits (known reference)
    pilot_bits = rng.integers(0, 2, size=n_pilots).astype(np.uint8)

    # Distributed pilot layout: first half at start, second half at end
    n_start = (n_pilots + 1) // 2
    n_end = n_pilots - n_start
    all_bits = np.concatenate([
        pilot_bits[:n_start], frame_bits, pilot_bits[n_start:]
    ])

    pilot_positions = list(range(n_start)) + list(
        range(n_symbols - n_end, n_symbols)
    )

    # NRZ encoding: 0 -> -1, 1 -> +1
    nrz = 2.0 * all_bits.astype(float) - 1.0

    # Upsample (rectangular pulse)
    nrz_up = np.repeat(nrz, sps)

    # Optional Gaussian filtering for GFSK
    if bt is not None:
        g = _gaussian_freq_filter(bt, sps)
        nrz_up = np.convolve(nrz_up, g, mode='same')

    # Phase accumulation
    n_samples = n_symbols * sps
    delta_phi = np.pi * h * nrz_up[:n_samples] / sps
    phi = np.cumsum(delta_phi)

    # Generate constant-envelope signal
    signal = np.exp(1j * phi)

    # Scale to target RMS (with small random variation)
    current_rms = np.sqrt(np.mean(np.abs(signal) ** 2))
    rms_scale = target_rms * (1.0 + 0.05 * rng.standard_normal())
    rms_scale = max(rms_scale, target_rms * 0.5)
    if current_rms > 0:
        signal *= rms_scale / current_rms

    # Apply CFO and random initial phase
    cfo = cfo_std * rng.standard_normal()
    phase0 = rng.uniform(0, 2 * np.pi)
    n_idx = np.arange(n_samples)
    phase_ramp = np.exp(1j * (2 * np.pi * cfo * n_idx + phase0))
    signal = signal * phase_ramp

    # Add AWGN
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
    )
    noisy = signal + noise

    # Build IQ tensor [1, 2, T]
    iq_tensor = np.zeros((1, 2, n_samples), dtype=np.float32)
    iq_tensor[0, 0, :] = noisy.real.astype(np.float32)
    iq_tensor[0, 1, :] = noisy.imag.astype(np.float32)

    return {
        'iq_tensor': iq_tensor,
        'iq_complex': noisy,
        'data_bits': data_bits,
        'frame_bits': frame_bits,
        'pilot_symbols': None,
        'pilot_bits': pilot_bits,
        'pilot_positions': pilot_positions,
        'n_pilots': n_pilots,
        'mod_type': mod,
        'snr_db': snr_db,
        'symbols_tx': None,
        'cfo': cfo,
        'phase0': phase0,
    }


# ---------------------------------------------------------------------------
# RX chain
# ---------------------------------------------------------------------------

def demodulate_burst(iq_complex, mod_type, n_pilots, pilot_symbols=None,
                     pilot_bits=None, sps=8, beta=0.35, pilot_positions=None,
                     iq_full=None, iq_win_start=None, n_guard=None):
    """
    Demodulate a burst. Dispatches to constellation or FSK demodulator.

    Args:
        iq_complex: Complex baseband samples (1D array, 128 samples)
        mod_type: Modulation type string
        n_pilots: Number of pilot symbols/bits
        pilot_symbols: Known pilot constellation points (constellation mods)
        pilot_bits: Known pilot bit values (FSK mods)
        sps: Samples per symbol
        beta: RRC roll-off factor (constellation mods only)
        pilot_positions: Symbol indices of pilots
        iq_full: Full signal with guard symbols (optional, for better demod)
        iq_win_start: Start index of data window in iq_full
        n_guard: Number of guard symbols

    Returns:
        dict: recovered_bits, data_bits, crc_pass, recovered_symbols
    """
    mod = _normalize_mod(mod_type)

    if mod in FSK_MODS:
        return _demodulate_burst_fsk(iq_complex, mod, n_pilots, pilot_bits,
                                     sps, pilot_positions)
    elif mod in CONSTELLATION_MODS:
        return _demodulate_burst_constellation(iq_complex, mod, n_pilots,
                                               pilot_symbols, sps, beta,
                                               pilot_positions,
                                               iq_full=iq_full,
                                               iq_win_start=iq_win_start,
                                               n_guard=n_guard)
    else:
        raise ValueError(f"Unknown modulation: {mod_type}")


def _demodulate_burst_constellation(iq_complex, mod, n_pilots, pilot_symbols,
                                    sps, beta, pilot_positions,
                                    iq_full=None, iq_win_start=None,
                                    n_guard=None):
    """Demodulate constellation-based modulations (BPSK..QAM64, PAM4).

    If iq_full is provided, applies the RX matched filter to the full
    signal (with guard context) and extracts the data symbols from the
    middle — this eliminates edge ISI that would otherwise corrupt
    high-order modulations like QAM64 on short bursts.
    """
    bps = get_bits_per_symbol(mod)

    iq = np.asarray(iq_complex, dtype=complex)
    n_samples = len(iq)
    n_sym_total = n_samples // sps

    # Default pilot positions: first at start, last at end
    if pilot_positions is None:
        if n_pilots == 1:
            pilot_positions = [0]
        else:
            n_start = (n_pilots + 1) // 2
            pilot_positions = list(range(n_start)) + list(
                range(n_sym_total - (n_pilots - n_start), n_sym_total)
            )

    pilot_positions = list(pilot_positions)
    pilot_symbols = np.asarray(pilot_symbols)

    if iq_full is not None and iq_win_start is not None and n_guard is not None:
        # --- Full-signal RX: matched filter on entire signal, extract middle ---
        iq_f = np.asarray(iq_full, dtype=complex)
        n_full_samples = len(iq_f)

        # RRC matched filter on full signal (no edge padding needed)
        num_taps = min(16 * sps + 1, n_full_samples // 2)
        if num_taps % 2 == 0:
            num_taps += 1
        h = rrc_filter(beta, sps, num_taps=num_taps)
        filtered_full = np.convolve(iq_f, h, mode='same')

        # Extract the data symbols from the guard region
        all_filtered_syms = filtered_full[0::sps]
        symbols_raw = all_filtered_syms[n_guard:n_guard + n_sym_total]
    else:
        # --- Window-only RX: pad edges and filter the 128-sample window ---
        num_taps = min(4 * sps + 1, n_samples // 2)
        if num_taps % 2 == 0:
            num_taps += 1
        h = rrc_filter(beta, sps, num_taps=num_taps)
        pad_len = num_taps // 2
        iq_padded = np.pad(iq, pad_len, mode='edge')
        filtered_padded = np.convolve(iq_padded, h, mode='same')
        filtered = filtered_padded[pad_len:pad_len + n_samples]
        symbols_raw = filtered[0::sps]

    if len(symbols_raw) < n_sym_total:
        symbols_raw = np.pad(symbols_raw, (0, n_sym_total - len(symbols_raw)))
    symbols_raw = symbols_raw[:n_sym_total]

    # 3. Pilot-based amplitude and phase recovery
    #    Using pilots gives correct scale even when the random symbol set
    #    has a different RMS than the full constellation (critical for QAM64).
    constellation = get_constellation(mod)
    rx_pilots = symbols_raw[pilot_positions]
    pos_arr = np.array(pilot_positions, dtype=float)

    # Amplitude: match pilot power to known pilot amplitude
    pilot_rx_rms = np.sqrt(np.mean(np.abs(rx_pilots) ** 2))
    pilot_tx_rms = np.sqrt(np.mean(np.abs(pilot_symbols) ** 2))
    if pilot_rx_rms > 1e-12:
        symbols_raw = symbols_raw * (pilot_tx_rms / pilot_rx_rms)
        rx_pilots = symbols_raw[pilot_positions]

    # Phase/CFO recovery using distributed pilot symbols
    if len(pilot_positions) >= 2 and (pos_arr[-1] - pos_arr[0]) > 0:
        pilot_phases = np.angle(rx_pilots * np.conj(pilot_symbols))
        pilot_phases_uw = np.unwrap(pilot_phases)
        coeffs = np.polyfit(pos_arr, pilot_phases_uw, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        sym_positions = np.arange(n_sym_total, dtype=float)
        correction = np.exp(-1j * (intercept + slope * sym_positions))
        symbols_corrected = symbols_raw * correction
    else:
        cross = np.sum(rx_pilots * np.conj(pilot_symbols))
        phase_est = np.angle(cross)
        symbols_corrected = symbols_raw * np.exp(-1j * phase_est)

    # 5. Extract data symbols (remove pilots)
    pilot_set = set(pilot_positions)
    data_symbols = np.array([
        symbols_corrected[i] for i in range(n_sym_total) if i not in pilot_set
    ])

    # 6. Hard decision and demap
    recovered_frame_bits = symbols_to_bits(data_symbols, mod)

    # 7. CRC check
    crc_pass = crc8_check(recovered_frame_bits)

    recovered_data_bits = (
        recovered_frame_bits[:-8] if len(recovered_frame_bits) > 8
        else np.array([], dtype=np.uint8)
    )

    return {
        'recovered_bits': recovered_frame_bits,
        'data_bits': recovered_data_bits,
        'crc_pass': crc_pass,
        'recovered_symbols': symbols_corrected,
    }


def _demodulate_burst_fsk(iq_complex, mod, n_pilots, pilot_bits,
                          sps, pilot_positions, h=0.5):
    """Demodulate FSK modulations (CPFSK, GFSK).

    Uses per-symbol endpoint phase change with pilot-based CFO estimation.
    No timing recovery needed (FSK symbols are at known sample boundaries).
    """
    iq = np.asarray(iq_complex, dtype=complex)
    n_samples = len(iq)
    n_sym_total = n_samples // sps

    # Default pilot positions
    if pilot_positions is None:
        if n_pilots == 1:
            pilot_positions = [0]
        else:
            n_start = (n_pilots + 1) // 2
            pilot_positions = list(range(n_start)) + list(
                range(n_sym_total - (n_pilots - n_start), n_sym_total))

    pilot_positions = list(pilot_positions)
    pilot_bits = np.asarray(pilot_bits, dtype=np.uint8)

    # 1. Per-symbol phase change (endpoint-to-endpoint)
    # phase_change[k] = angle(iq[(k+1)*sps] * conj(iq[k*sps]))
    # = pi*h*d_k + 2*pi*cfo*sps  (for full-length symbols)
    # Normalize to per-sample frequency for uniform handling
    sym_freq = np.zeros(n_sym_total)
    for k in range(n_sym_total):
        s0 = k * sps
        s1 = min((k + 1) * sps, n_samples - 1)
        span = s1 - s0
        if span > 0:
            sym_freq[k] = np.angle(iq[s1] * np.conj(iq[s0])) / span

    # 2. CFO estimation using pilot bits
    # sym_freq[k] ~ pi*h*d_k/sps + 2*pi*cfo
    pilot_nrz = 2.0 * pilot_bits.astype(float) - 1.0
    pilot_freqs = sym_freq[pilot_positions]
    expected_freq = np.pi * h * pilot_nrz / sps
    freq_offset = np.mean(pilot_freqs - expected_freq)

    # 3. Correct and threshold
    corrected_freq = sym_freq - freq_offset
    # Positive freq -> bit 1, negative -> bit 0
    all_bits = (corrected_freq > 0).astype(np.uint8)

    # 6. Extract data bits (remove pilots)
    pilot_set = set(pilot_positions)
    data_bit_list = [all_bits[i] for i in range(n_sym_total)
                     if i not in pilot_set]
    recovered_frame_bits = np.array(data_bit_list, dtype=np.uint8)

    # 7. CRC check
    crc_pass = crc8_check(recovered_frame_bits)

    recovered_data_bits = (
        recovered_frame_bits[:-8] if len(recovered_frame_bits) > 8
        else np.array([], dtype=np.uint8)
    )

    return {
        'recovered_bits': recovered_frame_bits,
        'data_bits': recovered_data_bits,
        'crc_pass': crc_pass,
        'recovered_symbols': None,
    }


# ---------------------------------------------------------------------------
# Convenience: generate + demodulate a batch
# ---------------------------------------------------------------------------

def generate_burst_batch(mod_type, n_bursts, n_symbols=16, n_pilots=2,
                         sps=8, beta=0.35, snr_db=18.0, target_rms=0.006,
                         cfo_std=0.015, seed=None):
    """
    Generate a batch of bursts.

    Returns:
        iq_batch: numpy array [N, 2, T]
        burst_info_list: list of dicts from generate_burst
    """
    rng = np.random.default_rng(seed)
    bursts = []
    for _ in range(n_bursts):
        burst = generate_burst(
            mod_type, n_symbols=n_symbols, n_pilots=n_pilots,
            sps=sps, beta=beta, snr_db=snr_db, target_rms=target_rms,
            cfo_std=cfo_std, rng=rng,
        )
        bursts.append(burst)

    iq_batch = np.concatenate([b['iq_tensor'] for b in bursts], axis=0)
    return iq_batch, bursts
