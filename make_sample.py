"""Generate a synthetic test tone clip so you can smoke-test the Acoustic tab without a mic.

Writes sample_tones.wav (a gliding vowel-like buzz with pauses). For real ASR testing,
record yourself or use any speech clip (LibriSpeech, Common Voice, your own voice).
"""
import numpy as np
import soundfile as sf

sr = 16000


def vowel(f0, formants, dur):
    t = np.arange(int(sr * dur)) / sr
    src = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 30))
    out = np.zeros_like(src)
    for f, bw in formants:
        # crude resonator: weight harmonics near each formant
        for k in range(1, 30):
            out += np.sin(2 * np.pi * f0 * k * t) / k * np.exp(-((f0 * k - f) / bw) ** 2)
    env = np.minimum(1, np.minimum(t, dur - t) / 0.05)
    return 0.4 * out / np.max(np.abs(out)) * env


gap = np.zeros(int(sr * 0.5))
clip = np.concatenate([
    vowel(120, [(730, 120), (1090, 150)], 0.8),   # /a/
    gap,
    vowel(130, [(270, 80), (2290, 200)], 0.8),    # /i/
    gap,
    vowel(110, [(300, 80), (870, 120)], 0.8),     # /u/
])
sf.write("sample_tones.wav", clip.astype(np.float32), sr)
print("wrote sample_tones.wav")
