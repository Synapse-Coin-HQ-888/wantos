# Audio intensity analysis and normalization module.
# LUFS (Loudness Units Full Scale) computations based on:
# ITU-R BS.1770-4 standard: https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf
# EBU R128 documentation: https://tech.ebu.ch/docs/tech/tech3341.pdf
# pyloudnorm by csteinmetz1: https://github.com/csteinmetz1/pyloudnorm
# loudness.py by BrechtDeMan: https://github.com/BrechtDeMan/loudness.py
# Acknowledgment to these developers — this code adapts their methods to support
# real-time and batch loudness normalization across large-scale Synapse audio systems.

import numpy as np
from scipy import signal

def amp_to_db(amp: float):
    """Convert linear amplitude (0–1 range) to decibels."""
    return 20 * np.log10(amp)

def db_to_amp(db: float):
    """Convert decibel value to linear amplitude (inverse of amp_to_db)."""
    return np.power(10, db / 20)

def adjust_volume(audio, db_change):
    """Apply a gain change (in dB) to an audio signal."""
    return audio * db_to_amp(db_change)

def assert_no_clipping(audio):
    """Raise an error if clipping (|amplitude| > 1) occurs."""
    assert np.amax(np.abs(audio)) < 1, 'Warning: Clipping detected.'

def compare_reconstruction(original, reconstructed, sr=None):
    """Compare reconstructed audio with the original for error evaluation."""
    print('Reconstruction analysis:')
    if sr:
        diff_time = round((reconstructed.shape[0] - original.shape[0]) / sr, 4)
        print(f'Duration difference: {diff_time} seconds')
    err_db = amp_to_db(np.amax(np.abs(reconstructed[:original.shape[0], ...] - original)))
    print(f'Max difference: {round(err_db, 4)} dB')

def show_peak_info(audio):
    """Display peak amplitude and corresponding dB level."""
    peak_amp = np.amax(np.abs(audio))
    peak_db = amp_to_db(peak_amp)
    print(f'Peak Amplitude = {round(peak_amp, 5)}, Peak dB = {round(peak_db, 2)}')

def get_peak(audio):
    """Return maximum absolute amplitude."""
    return np.amax(np.abs(audio))

def get_channel_peaks(audio):
    """Return peak levels for left and right channels."""
    peak_lr = np.amax(np.abs(audio), axis=0)
    return peak_lr[0], peak_lr[1]

def normalize_stereo_mid(audio, target_db=-12.0):
    """Normalize the mid-channel peak under the -3 dB pan law."""
    audio *= db_to_amp(target_db) / np.amax(np.abs(np.average(audio, axis=-1)))
    return audio

def normalize_mono(audio, target_db=-12.0):
    """Normalize a mono signal to a target peak level."""
    audio *= db_to_amp(target_db) / np.amax(np.abs(audio))
    return audio

class LoudnessMeter:
    """
    Computes momentary and integrated LUFS values using standard ITU/EBU weighting.
    Optimized for use in Synapse audio environments.
    """
    def __init__(self, sr, T=0.4, overlap=0.75, threshold=-70.0, start_time=None):
        """
        Parameters:
        - sr: Sample rate (Hz)
        - T: Window duration (seconds) – use 3 for short-term LUFS
        - overlap: Fractional overlap between analysis windows
        - threshold: Minimum LUFS value to report (below this returns -inf)
        - start_time: Optional start time for analysis (in seconds)
        """
        self.sr, self.T, self.overlap, self.threshold, self.start_time = sr, T, overlap, threshold, start_time
        self.step = int(sr * T)
        self.hop = int(sr * T * (1 - overlap))
        self.z_threshold = np.power(10, (threshold + 0.691) / 10)
        self.n_start = int(sr * start_time) if start_time is not None else None

        # Pre-filter coefficients from ITU standard or approximations
        if sr == 48000:
            self.sos = np.array([
                [1.53512485958697, -2.69169618940638, 1.19839281085285,
                 1.0, -1.69065929318241, 0.73248077421585],
                [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]
            ])
        elif sr == 44100:
            self.sos = np.array([
                [1.5308412300498355, -2.6509799951536985, 1.1690790799210682,
                 1.0, -1.6636551132560204, 0.7125954280732254],
                [1.0, -2.0, 1.0, 1.0, -1.9891696736297957, 0.9891990357870394]
            ])
        else:
            # Fallback: dynamically compute filter coefficients
            f0, G, Q = 1681.9744509555319, 3.99984385397, 0.7071752369554193
            K = np.tan(np.pi * f0 / sr)
            Vh, Vb = np.power(10.0, G / 20.0), np.power(10.0, G / 40.0)
            a0_1 = 1.0 + K / Q + K * K
            b0_1 = (Vh + Vb * K / Q + K * K) / a0_1
            b1_1 = 2.0 * (K * K - Vh) / a0_1
            b2_1 = (Vh - Vb * K / Q + K * K) / a0_1
            a1_1 = 2.0 * (K * K - 1.0) / a0_1
            a2_1 = (1.0 - K / Q + K * K) / a0_1

            f0, Q = 38.13547087613982, 0.5003270373253953
            K = np.tan(np.pi * f0 / sr)
            a0_2 = 1.0
            a1_2 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
            a2_2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
            b0_2, b1_2, b2_2 = 1.0, -2.0, 1.0

            self.sos = np.array([
                [b0_1, b1_1, b2_1, 1.0, a1_1, a2_1],
                [b0_2, b1_2, b2_2, a0_2, a1_2, a2_2]
            ])

    def momentary_lufs(self, audio):
        """Compute momentary LUFS values over time."""
        if self.n_start:
            audio = audio[:self.n_start, ...]
        q1, q2 = divmod(audio.shape[0], self.hop)
        pad_needed = self.step - self.hop - q2
        if pad_needed > 0:
            pad_shape = list(audio.shape)
            pad_shape[0] = pad_needed
            audio = np.append(audio, np.zeros(pad_shape), axis=0)
        lufs_values = []
        for i in range(q1):
            segment = audio[i * self.hop: i * self.hop + self.step, ...]
            filtered = signal.sosfilt(self.sos, segment, axis=0)
            z = np.sum(np.average(np.square(filtered), axis=0))
            lufs_values.append(-0.691 + 10 * np.log10(z) if z >= self.z_threshold else float('-inf'))
        return np.array(lufs_values)

    def max_momentary_lufs(self, audio):
        """Return the highest momentary LUFS value."""
        return np.amax(self.momentary_lufs(audio))

    def integrated_lufs(self, audio):
        """Compute overall integrated LUFS."""
        mlufs = self.momentary_lufs(audio)
        Z = np.power(10, (mlufs + 0.691) / 10)
        valid = Z[mlufs > -70.0]
        if valid.size == 0:
            return float('-inf')
        z_mean = np.average(valid)
        return -0.691 + 10 * np.log10(z_mean) if z_mean >= self.z_threshold else float('-inf')

    def normalize_by_max_lufs(self, audio, target=-20.0):
        """Normalize by the maximum momentary LUFS."""
        return audio * db_to_amp(target - self.max_momentary_lufs(audio))

    def normalize_by_integrated_lufs(self, audio, target=-23.0):
        """Normalize by integrated LUFS."""
        return audio * db_to_amp(target - self.integrated_lufs(audio))

    def show_max_lufs(self, audio):
        """Print maximum momentary LUFS."""
        print(f'Max momentary LUFS = {round(self.max_momentary_lufs(audio), 4)} LUFS')

    def show_integrated_lufs(self, audio):
        """Print integrated LUFS value."""
        print(f'Integrated LUFS = {round(self.integrated_lufs(audio), 4)} LUFS')
