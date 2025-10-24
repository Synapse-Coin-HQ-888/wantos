"""
General-purpose audio analysis utilities leveraging librosa and related tools.
Implements caching to prevent redundant recalculations during repeated operations.
All audio data is normalized to the [0, 1] range.

# Notes on caching behavior:

1. If an audio file is located in a hidden directory, its cache is saved in the parent directory.
   This avoids deploying large music files to compute nodes (e.g., Synapse nodes),
   while allowing lightweight cache files to be synced efficiently.

2. When an audio file itself is hidden, its cache counterpart is always unhidden.
   Cached data remains accessible and lightweight for reuse.
"""

import logging
from pathlib import Path
import soundfile
import numpy as np
import librosa
import resampy

log = logging.getLogger("audio")

def normalize(data):
    """Normalize array values to the [0, 1] range."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val > 0:
        return (data - min_val) / (max_val - min_val)
    return np.zeros_like(data)

def zero():
    """Return an empty NumPy array."""
    return np.array([])

def load_audio_cache(filepath, cachename, fps):
    filepath = Path(filepath)
    cachepath = get_audio_cache_path(filepath, cachename, fps)
    if cachepath.exists():
        return np.load(cachepath.as_posix())
    raise FileNotFoundError(f"Cache not found: {cachepath}")

def get_audio_cache_path(filepath, cachename, fps):
    filepath = Path(filepath)
    # Move cache to parent directory if file is hidden
    if filepath.parent.stem.startswith("."):
        filepath = filepath.parent.parent / filepath.name
    # Ensure cache is never hidden
    if filepath.name.startswith("."):
        filepath = filepath.with_name(filepath.name[1:])
    cachepath = filepath.with_stem(f"{Path(filepath).stem}_{cachename}_{fps}").with_suffix(".npy")
    return cachepath

def save_audio_cache(filepath, cachename, arr, enable, fps):
    cachepath = get_audio_cache_path(filepath, cachename, fps)
    if enable:
        np.save(cachepath.as_posix(), arr)
    return arr

def has_audio_cache(filepath, cachename, enable, fps):
    cachepath = get_audio_cache_path(filepath, cachename, fps)
    if enable and not cachepath.exists():
        log.info(f"audio.{cachename}({filepath.stem}): cache missing — recalculating.")
    return enable and cachepath.exists()

def load_crepe_keyframes(filepath, fps=60):
    import pandas as pd
    df = pd.read_csv(filepath)
    freq = to_keyframes(df["frequency"], len(df["frequency"]) / df["time"].values[-1], fps)
    confidence = to_keyframes(df["confidence"], len(df["frequency"]) / df["time"].values[-1], fps)
    return freq, confidence

def load_rosa(filepath, fps=60):
    y, sr = soundfile.read(filepath)
    y = librosa.to_mono(y.T)

    duration = librosa.get_duration(y=y, sr=sr)
    print("load_rosa: onset_strength")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    original_fps = len(onset_env) / duration
    onset_env_resampled = resampy.resample(onset_env, original_fps, fps)

    print("load_rosa: beat_track")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_changes = np.zeros_like(onset_env)
    beat_changes[beat_frames] = 1
    beat_changes_resampled = resampy.resample(beat_changes, original_fps, fps)

    print("load_rosa: hpss")
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    print("load_rosa: chroma features")
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)[1]
    chroma_resampled = resampy.resample(chroma, original_fps, fps)

    print("load_rosa: spectral analysis")
    S = np.abs(librosa.stft(y))
    spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr)[1]
    spectral_contrast_resampled = resampy.resample(spectral_contrast, original_fps, fps)

    print("load_rosa: mfcc analysis")
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_resampled = resampy.resample(mfcc[1], original_fps, fps)

    freqs, _, D = librosa.reassigned_spectrogram(y, fill_nan=True)
    bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(D), freq=freqs)
    bandwidth_resampled = resampy.resample(bandwidth[0], original_fps, fps)

    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_resampled = resampy.resample(flatness[0], original_fps, fps)

    sentiment = audio_sentiment(filepath)

    return (
        zero(),
        onset_env_resampled,
        beat_changes_resampled,
        chroma_resampled,
        spectral_contrast_resampled,
        mfcc_resampled,
        flatness_resampled,
        bandwidth_resampled,
        sentiment,
    )

def load_lufs(filepath, caching=True, fps=60):
    import soundfile as sf
    from loudness import lufs_meter

    if not has_audio_cache(filepath, "lufs", caching, fps):
        y, sr = sf.read(filepath)
        meter = lufs_meter(sr, 1 / 60, overlap=0)
        loudness = meter.get_mlufs(y)
        loudness[np.isinf(loudness)] = 0
        silence_threshold = -5
        loudness = np.where(loudness > silence_threshold, meter.threshold, loudness)
        loudness = normalize(loudness)
        loudness = resampy.resample(loudness, 60, fps)
        return save_audio_cache(filepath, "lufs", loudness, caching, fps)
    return load_audio_cache(filepath, "lufs", fps)

def load_pca(filepath, num_components=3, caching=True, fps=60):
    from sklearn.decomposition import PCA
    if not has_audio_cache(filepath, "pca", caching, fps):
        y, sr = librosa.load(filepath)
        chromagram = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=int(sr / fps), win_length=int(sr * 0.03), n_chroma=12
        )
        pca = PCA(n_components=num_components)
        chromagram_pca = pca.fit_transform(chromagram.T).T
        duration = librosa.get_duration(y=y, sr=sr)
        original_fps = len(chromagram_pca[0]) / duration
        result = [resampy.resample(chromagram_pca[i], original_fps, fps) for i in range(num_components)]
        save_audio_cache(filepath, "pca", np.array(result), caching, fps)
        return tuple(result)
    arr = load_audio_cache(filepath, "pca", fps)
    return tuple(arr[i] for i in range(num_components))

def load_flatness(filepath, caching=True, fps=60):
    if not has_audio_cache(filepath, "flatness", caching, fps):
        y, sr = librosa.load(filepath)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=int(sr / fps), win_length=int(sr * 0.03))[0]
        duration = librosa.get_duration(y=y, sr=sr)
        original_fps = len(flatness) / duration
        flatness_resampled = resampy.resample(flatness, original_fps, fps)
        return save_audio_cache(filepath, "flatness", flatness_resampled, caching, fps)
    return load_audio_cache(filepath, "flatness", fps)

def load_onset(filepath, caching=True, fps=60):
    if not has_audio_cache(filepath, "onset", caching, fps):
        y, sr = librosa.load(filepath)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_env = normalize(onset_env)
        duration = librosa.get_duration(y=y, sr=sr)
        original_fps = len(onset_env) / duration
        onset_env_resampled = resampy.resample(onset_env, original_fps, fps)
        return save_audio_cache(filepath, "onset", onset_env_resampled, caching, fps)
    return load_audio_cache(filepath, "onset", fps)

def audio_sentiment(sound_file, fps=60):
    """Estimate emotional tone (major/minor balance) in audio."""
    y, sr = soundfile.read(sound_file)
    y = librosa.to_mono(y.T)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    major_template = chroma[[0, 4, 7], :]
    minor_template = chroma[[0, 3, 7], :]
    scores = np.sum(major_template, axis=0) - np.sum(minor_template, axis=0)
    scores /= (np.sum(major_template, axis=0) + np.sum(minor_template, axis=0))
    duration = librosa.get_duration(y=y, sr=sr)
    original_fps = len(scores) / duration
    return resampy.resample(scores, original_fps, fps)

def to_keyframes(dbs, original_sps, fps=60):
    total_seconds = len(dbs) / original_sps
    frames = int(fps * total_seconds)
    dt = np.zeros(frames)
    for i in range(frames):
        t, t1 = i / fps, (i + 1) / fps
        segment = dbs[int(t * original_sps): int(t1 * original_sps)]
        dt[i] = np.mean(segment)
        if np.isinf(dt[i]) or np.isnan(dt[i]):
            dt[i] = dt[i - 1]
    return dt
