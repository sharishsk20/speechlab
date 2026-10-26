"""Acoustic analysis: waveform, spectrogram, MFCC, pitch (F0), energy, pauses."""
import numpy as np
import librosa

SR = 16000


def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def spectrogram_db(y, sr=SR, n_fft=1024, hop=256):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    return librosa.amplitude_to_db(S, ref=np.max), hop


def mfcc(y, sr=SR, n_mfcc=13, hop=256):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop)


def pitch_track(y, sr=SR, hop=256):
    """F0 contour via pYIN. Returns (times, f0) with NaN for unvoiced frames."""
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sr, frame_length=1024, hop_length=hop,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    return times, f0


def rms_energy(y, hop=256):
    return librosa.feature.rms(y=y, hop_length=hop)[0]


def find_pauses(y, sr=SR, top_db=35, min_pause=0.3):
    """Return list of (start_s, end_s) silent gaps longer than min_pause."""
    intervals = librosa.effects.split(y, top_db=top_db)
    pauses = []
    for (_, e1), (s2, _) in zip(intervals[:-1], intervals[1:]):
        gap = (s2 - e1) / sr
        if gap >= min_pause:
            pauses.append((e1 / sr, s2 / sr))
    return pauses


def summarize(y, sr=SR):
    """Headline acoustic statistics for the Acoustic tab."""
    times, f0 = pitch_track(y, sr)
    voiced = f0[~np.isnan(f0)]
    pauses = find_pauses(y, sr)
    dur = len(y) / sr
    return {
        "duration_s": dur,
        "mean_f0_hz": float(np.mean(voiced)) if len(voiced) else 0.0,
        "f0_range_hz": (float(np.min(voiced)), float(np.max(voiced))) if len(voiced) else (0.0, 0.0),
        "voiced_ratio": float(len(voiced) / max(len(f0), 1)),
        "n_pauses": len(pauses),
        "pause_time_s": float(sum(e - s for s, e in pauses)),
        "rms_mean": float(np.mean(rms_energy(y))),
    }
