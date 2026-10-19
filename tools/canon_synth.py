# Renders Pachelbel's Canon in D (public-domain composition) as a soft piano loop.
import numpy as np, wave, sys
SR = 32000
Q = 60 / 62            # quarter note in seconds (62 bpm)
CYC = 8 * Q            # one pass of the ground bass
N_CYC = 12
rng = np.random.default_rng(7)

def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)

def piano(m, dur, vel):
    f0 = mtof(m)
    ring = min(4.5, dur + 2.8)
    n = int(ring * SR); t = np.arange(n) / SR
    out = np.zeros(n)
    base = 0.45 + f0 / 700
    for k in range(1, 9):
        fk = f0 * k * np.sqrt(1 + 0.0004 * k * k)
        if fk > SR / 2.2: break
        amp = (1 / k ** 1.25) * (0.9 if k == 2 else 1)
        out += amp * np.sin(2 * np.pi * fk * t + rng.random() * 6.28) * np.exp(-t * base * (1 + 0.55 * (k - 1)))
    att = np.minimum(1, t / 0.006)
    rel_start = dur + 0.25
    rel = np.where(t < rel_start, 1, np.exp(-(t - rel_start) * 6))
    return out * att * rel * vel

notes = []  # (start, midi, dur, vel)
def add(start, m, dur, vel): notes.append((start, m, dur, vel))

bass   = [50, 45, 47, 42, 43, 38, 43, 45]                      # D A B F# G D G A
chords = [[62, 66, 69], [61, 64, 69], [59, 62, 66], [61, 66, 69],
          [59, 62, 67], [62, 66, 69], [59, 62, 67], [61, 64, 69]]
melA = [78, 76, 74, 73, 71, 69, 71, 73]                        # F# E D C# B A B C#
melB = [74, 73, 71, 69, 67, 66, 67, 64]                        # D C# B A G F# G E
eighths = [74, 78, 81, 79, 78, 74, 78, 76, 74, 71, 74, 81, 79, 83, 81, 79,
           78, 74, 76, 85, 86, 90, 93, 81, 83, 79, 81, 78, 74, 86, 86, 85]

for c in range(N_CYC):
    t0 = c * CYC
    # gentle swell over the piece, softer at both ends so the loop joins smoothly
    env = 0.55 + 0.45 * np.sin(np.pi * (c + 0.5) / N_CYC)
    for i in range(8):
        tb = t0 + i * Q
        add(tb, bass[i], Q * 0.95, 0.42 * env)
        add(tb, bass[i] - 12, Q * 0.95, 0.20 * env)
        ch = chords[i]
        # broken chord in eighths
        add(tb + Q * 0.5, ch[1], Q * 0.45, 0.16 * env)
        add(tb + Q * 0.75 if c >= 6 else tb + Q * 0.5 + 0.001, ch[2], Q * 0.45, 0.13 * env)
    phase = c
    if phase in (2, 3, 10):
        for i, m in enumerate(melA): add(t0 + i * Q, m, Q * 0.98, 0.34 * env)
    if phase in (4, 5, 11):
        for i, m in enumerate(melB): add(t0 + i * Q, m, Q * 0.98, 0.32 * env)
        if phase != 11:
            for i, m in enumerate(melA): add(t0 + i * Q, m, Q * 0.98, 0.22 * env)
    if phase in (6, 7, 8, 9):
        half = (phase % 2) * 16
        for j in range(16):
            add(t0 + j * Q / 2, eighths[half + j], Q * 0.5, 0.27 * env)
        for i, m in enumerate(melB): add(t0 + i * Q, m - (12 if phase in (8, 9) else 0), Q * 0.98, 0.2 * env)

total = N_CYC * CYC
tail = 4.0
L = int((total + tail) * SR)
left = np.zeros(L); right = np.zeros(L)
for start, m, dur, vel in notes:
    s = piano(m, dur, vel)
    i0 = int(start * SR); i1 = min(L, i0 + len(s)); s = s[: i1 - i0]
    pan = np.clip((m - 60) / 40, -0.5, 0.5)
    left[i0:i1] += s * (0.5 - pan * 0.6)
    right[i0:i1] += s * (0.5 + pan * 0.6)

# hall reverb: decaying filtered noise, a different one per channel
def ir(seed):
    r = np.random.default_rng(seed)
    n = int(3.2 * SR); t = np.arange(n) / SR
    x = r.standard_normal(n) * np.exp(-t / 0.75)
    k = 6; x = np.convolve(x, np.ones(k) / k, mode="same")   # soften highs
    x[: int(0.012 * SR)] = 0
    return x / np.sqrt(np.sum(x ** 2))
def conv(a, b):
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    return np.fft.irfft(np.fft.rfft(a, n) * np.fft.rfft(b, n), n)[: len(a)]
wl = conv(left, ir(1)); wr = conv(right, ir(2))
left = left * 0.78 + wl * 0.42
right = right * 0.78 + wr * 0.42

# fold the tail onto the start so the loop is seamless
n_tot = int(total * SR)
for ch in (left, right):
    ch[: L - n_tot] += ch[n_tot:]
left = left[:n_tot]; right = right[:n_tot]
st = np.stack([left, right], axis=1)
st *= 0.89 / np.max(np.abs(st))
pcm = (st * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("seconds", round(total, 1), "notes", len(notes))
