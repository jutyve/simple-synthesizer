import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile

# ============================
# CONFIG
# ============================

SONG_NAME = "test"
SAMPLES_FOLDER = "tunes"      # folder with key samples: a.wav, w.wav, ..., ;.wav
KEY_ORDER = ['a','w','s','e','d','f','t','g','y','h','u','j','k','o','l','p',';']
HOP_LENGTH = 512
SR = 44100                    # target sample rate
FMIN = 80
FMAX = 1000
VOLUME = 0.9                  # reduce slightly to avoid clipping

# ============================
# Load samples
# ============================

samples = {}
sample_lengths = {}

for key in KEY_ORDER:
    path = os.path.join(SAMPLES_FOLDER, f"{key}.wav")
    if os.path.exists(path):
        y, sr_sample = librosa.load(path, sr=SR)  # resample everything to SR
        samples[key] = y
        sample_lengths[key] = len(y)
    else:
        print(f"Warning: missing sample for key {key}")

print(f"Loaded {len(samples)} samples")

# ============================
# Load input audio
# ============================

y, sr_orig = librosa.load(f"{SONG_NAME}.wav", sr=SR)

# ============================
# Onset detection
# ============================

onset_frames = librosa.onset.onset_detect(
    y=y,
    sr=SR,
    hop_length=HOP_LENGTH,
    units='frames'
)
onset_samples = librosa.frames_to_samples(
    onset_frames,
    hop_length=HOP_LENGTH
)

# ============================
# Pitch detection helper
# ============================

def pitch_to_key_index(freq, min_freq=FMIN, max_freq=FMAX, n_keys=len(KEY_ORDER)):
    freq = max(min_freq, min(max_freq, freq))
    # invert mapping: lower freq -> higher key
    key_idx = int(round((max_freq - freq) / (max_freq - min_freq) * (n_keys - 1)))
    return min(n_keys - 1, max(0, key_idx))

# ============================
# Build output
# ============================

output = np.zeros(len(y))

for onset_sample in onset_samples:
    # take a small window to estimate pitch
    window = y[onset_sample : onset_sample + HOP_LENGTH]
    if len(window) == 0:
        continue

    # detect pitch using YIN
    f0 = librosa.yin(
        window,
        fmin=FMIN,
        fmax=FMAX,
        sr=SR,
        frame_length=len(window),
        hop_length=len(window)
    )

    if len(f0) == 0 or np.isnan(f0[0]) or f0[0] <= 0:
        continue

    freq = f0[0]
    key_idx = pitch_to_key_index(freq)
    key = KEY_ORDER[key_idx]

    if key not in samples:
        continue

    sample = samples[key]
    sample_len = sample_lengths[key]

    end = onset_sample + sample_len
    if end > len(output):
        sample = sample[:len(output) - onset_sample]
        end = len(output)

    output[onset_sample:end] += sample

# ============================
# Normalize & save
# ============================

max_val = np.max(np.abs(output))
if max_val > 0:
    output = (output / max_val) * VOLUME

wavfile.write("output_keyboard_onsets.wav",SR, output.astype(np.float32))
print("Saved as output_keyboard_onsets.wav")
