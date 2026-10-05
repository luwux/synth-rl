"""Timbre descriptors and spectral distances computed from rendered audio (numpy only)."""

from __future__ import annotations

import numpy as np

from .vital_env import NOTE_DUR, SR

EPS = 1e-9
HOP = 512
WIN = 2048


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def _frames(x: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _stft_mag(x: np.ndarray, win: int = WIN, hop: int = HOP) -> np.ndarray:
    return np.abs(np.fft.rfft(_frames(x, win, hop) * np.hanning(win), axis=1))


def describe(audio: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Descriptors used by edit rewards.

    centroid_oct: log2 of the energy-weighted spectral centroid (Hz) while the note is held.
    hf_db: energy above 2 kHz relative to total, in dB.
    loudness_db: peak frame RMS level, so envelope edits (sustain, attack) do not count as level changes.
    attack_s: time from onset to 90% of peak frame RMS.
    sustain: frame RMS just before note-off divided by peak RMS.
    tail_s: time after note-off until frame RMS falls 40 dB below peak.
    width: side energy / (mid + side) energy.
    """
    mono = _mono(audio)
    rms = np.sqrt((_frames(mono) ** 2).mean(axis=1)) + EPS
    t = (np.arange(len(rms)) * HOP + WIN / 2) / sr
    peak = rms.max()
    held = t < NOTE_DUR
    floor = peak * 10 ** (-40 / 20)

    mag = _stft_mag(mono)
    freqs = np.fft.rfftfreq(WIN, 1 / sr)
    power = mag**2
    hp = power[held] if held.any() else power
    centroid = float((hp * freqs).sum() / (hp.sum() + EPS))
    hf = float(hp[:, freqs >= 2000].sum() / (hp.sum() + EPS))

    onset_idx = int(np.argmax(rms > floor))
    attack_idx = int(np.argmax(rms >= 0.9 * peak))
    attack = max(0.0, (attack_idx - onset_idx) * HOP / sr)

    off_idx = int(np.searchsorted(t, NOTE_DUR)) - 1
    sustain = float(rms[max(off_idx - 2, 0)] / peak)
    after = np.where((t > NOTE_DUR) & (rms < floor))[0]
    tail = float(t[after[0]] - NOTE_DUR) if len(after) else float(t[-1] - NOTE_DUR)

    if audio.ndim == 2 and audio.shape[0] == 2:
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        width = float((side**2).sum() / ((mid**2).sum() + (side**2).sum() + EPS))
    else:
        width = 0.0

    return {
        "centroid_oct": float(np.log2(centroid + 1.0)),
        "hf_db": float(10 * np.log10(hf + EPS)),
        "loudness_db": float(20 * np.log10(peak)),
        "attack_s": attack,
        "sustain": sustain,
        "tail_s": tail,
        "width": width,
    }


def mss_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Multi-resolution log-magnitude STFT L1 plus spectral convergence, on the mono mix."""
    a, b = _mono(a), _mono(b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    total = 0.0
    for win in (512, 1024, 2048):
        hop = win // 4
        ma, mb = _stft_mag(a, win, hop), _stft_mag(b, win, hop)
        log_l1 = np.abs(np.log(ma + 1e-5) - np.log(mb + 1e-5)).mean()
        conv = np.linalg.norm(ma - mb) / (np.linalg.norm(mb) + EPS)
        total += float(log_l1 + conv)
    return total / 3
