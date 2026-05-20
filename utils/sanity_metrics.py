import numpy as np
import torch

def exponential_moving_average(data, alpha=0.1):
    """Compute EMA of a 1‑D numpy array or torch tensor.
    Returns the same type as input.
    """
    if isinstance(data, torch.Tensor):
        ema = torch.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema
    else:
        ema = np.empty_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

def autocorrelation(x, max_lag=None):
    """Return autocorrelation of 1‑D signal `x` for lags up to `max_lag`.
    Uses FFT‑based convolution for speed.
    """
    x = np.asarray(x)
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    x_centered = x - np.mean(x)
    corr = np.fft.ifft(np.fft.fft(x_centered, n*2) * np.conj(np.fft.fft(x_centered, n*2))).real[:n]
    corr /= corr[0]
    return corr[:max_lag]

def steady_state_check(observables, tolerance=0.01, window_frac=0.1):
    """Determine if a time series has reached steady state.
    * `observables` – 1‑D array of scalar measurements.
    * `tolerance` – relative difference allowed between last window mean and overall mean.
    * `window_frac` – fraction of the series to consider as the final window.
    Returns True/False and the relative difference.
    """
    obs = np.asarray(observables)
    overall_mean = np.mean(obs)
    window_len = int(len(obs) * window_frac)
    final_mean = np.mean(obs[-window_len:])
    rel_diff = np.abs(final_mean - overall_mean) / (np.abs(overall_mean) + 1e-12)
    return rel_diff < tolerance, rel_diff
