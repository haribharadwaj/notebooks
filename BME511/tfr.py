"""A module for time-frequency estimation.

Authors : Hari Bharadwaj <hari.bharadwaj@gmail.com>

License : BSD 3-clause

Multitaper wavelet method adapted/isolated for teaching from
Hari Bharadwaj's ANLffr toolbox. This code was also added to
the MNE-Python package.
"""

import warnings
from math import sqrt
import numpy as np
from scipy import linalg
from scipy.fftpack import fftn, ifftn
from scipy.signal.windows import dpss


def _dpss_wavelet(sfreq, freqs, n_cycles=7, time_bandwidth=4.0,
                  zero_mean=False):
    """Compute Wavelets for the given frequency range

    Parameters
    ----------
    sfreq : float
        Sampling Frequency.
    freqs : ndarray, shape (n_freqs,)
        The frequencies in Hz.
    n_cycles : float | ndarray, shape (n_freqs,)
        The number of cycles globally or for each frequency.
        Defaults to 7.
    time_bandwidth : float, (optional)
        Time x Bandwidth product.
        The number of good tapers (low-bias) is chosen automatically based on
        this to equal floor(time_bandwidth - 1).
        Default is 4.0, giving 3 good tapers.

    Returns
    -------
    Ws : list of array
        Wavelets time series
    """
    Ws = list()
    if time_bandwidth < 2.0:
        raise ValueError("time_bandwidth should be >= 2.0 for good tapers")
    n_taps = int(np.floor(time_bandwidth - 1))
    n_cycles = np.atleast_1d(n_cycles)

    if n_cycles.size != 1 and n_cycles.size != len(freqs):
        raise ValueError("n_cycles should be fixed or defined for "
                         "each frequency.")

    for m in range(n_taps):
        Wm = list()
        for k, f in enumerate(freqs):
            if len(n_cycles) != 1:
                this_n_cycles = n_cycles[k]
            else:
                this_n_cycles = n_cycles[0]

            t_win = this_n_cycles / float(f)
            t = np.arange(0., t_win, 1.0 / sfreq)
            # Making sure wavelets are centered before tapering
            oscillation = np.exp(2.0 * 1j * np.pi * f * (t - t_win / 2.))

            # Get dpss tapers
            tapers, conc = dpss(t.shape[0], time_bandwidth / 2., n_taps,
                return_ratios=True)

            Wk = oscillation * tapers[m]
            if zero_mean:  # to make it zero mean
                real_offset = Wk.mean()
                Wk -= real_offset
            Wk /= sqrt(0.5) * linalg.norm(Wk.ravel())

            Wm.append(Wk)

        Ws.append(Wm)

    return Ws



def _centered(arr, newsize):
    """Aux Function to center data"""
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]



def _cwt_fft(X, Ws, mode="same"):
    """Compute cwt with fft based convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    # Precompute wavelets for given frequency range to save time
    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    Ws_max_size = max(W.size for W in Ws)
    size = n_times + Ws_max_size - 1
    # Always use 2**n-sized FFT
    fsize = 2 ** int(np.ceil(np.log2(size)))

    # precompute FFTs of Ws
    fft_Ws = np.empty((n_freqs, fsize), dtype=np.complex128)
    for i, W in enumerate(Ws):
        if len(W) > n_times:
            raise ValueError('Wavelet is too long for such a short signal. '
                             'Reduce the number of cycles.')
        fft_Ws[i] = fftn(W, [fsize])

    for k, x in enumerate(X):
        if mode == "full":
            tfr = np.zeros((n_freqs, fsize), dtype=np.complex128)
        elif mode == "same" or mode == "valid":
            tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)

        fft_x = fftn(x, [fsize])
        for i, W in enumerate(Ws):
            ret = ifftn(fft_x * fft_Ws[i])[:n_times + W.size - 1]
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = _centered(ret, sz)
            else:
                tfr[i, :] = _centered(ret, n_times)
        yield tfr



def _cwt_convolve(X, Ws, mode='same'):
    """Compute time freq decomposition with temporal convolutions
    Return a generator over signals.
    """
    X = np.asarray(X)

    n_signals, n_times = X.shape
    n_freqs = len(Ws)

    # Compute convolutions
    for x in X:
        tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)
        for i, W in enumerate(Ws):
            ret = np.convolve(x, W, mode=mode)
            if len(W) > len(x):
                raise ValueError('Wavelet is too long for such a short '
                                 'signal. Reduce the number of cycles.')
            if mode == "valid":
                sz = abs(W.size - n_times) + 1
                offset = (n_times - sz) / 2
                tfr[i, offset:(offset + sz)] = ret
            else:
                tfr[i] = ret
        yield tfr



def _time_frequency(X, Ws, use_fft, decim):
    """Aux of time_frequency for parallel computing over channels
    """
    n_trials, n_times = X.shape
    n_times = n_times // decim + bool(n_times % decim)
    n_frequencies = len(Ws)
    psd = np.zeros((n_frequencies, n_times))  # PSD
    plf = np.zeros((n_frequencies, n_times), np.complex)  # phase lock

    mode = 'same'
    if use_fft:
        tfrs = _cwt_fft(X, Ws, mode)
    else:
        tfrs = _cwt_convolve(X, Ws, mode)

    for tfr in tfrs:
        tfr = tfr[:, ::decim]
        tfr_abs = np.abs(tfr)
        psd += tfr_abs ** 2
        plf += tfr / tfr_abs
    psd /= n_trials
    plf = np.abs(plf) / n_trials
    return psd, plf



def tfr_multitaper(data, sfreq, frequencies, time_bandwidth=4.0,
                   use_fft=True, n_cycles=7, decim=1,
                   zero_mean=True, return_itc=False,
                   verbose=None):
    """Compute time-frequency power and inter-trial coherence

    The time-frequency analysis is done with DPSS wavelets

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_times)
        The input data.
    sfreq : float
        sampling Frequency
    frequencies : np.ndarray, shape (n_frequencies,)
        Array of frequencies of interest
    time_bandwidth : float
        Time x (Full) Bandwidth product.
        The number of good tapers (low-bias) is chosen automatically based on
        this to equal floor(time_bandwidth - 1). Default is 4.0 (3 tapers).
    use_fft : bool
        Compute transform with fft based convolutions or temporal
        convolutions. Defaults to True.
    n_cycles : float | np.ndarray shape (n_frequencies,)
        Number of cycles. Fixed number or one per frequency. Defaults to 7.
    decim: int
        Temporal decimation factor. Defaults to 1.
    return_itc: bool
        Return the inter-trial coherence. Defaults to False.
    zero_mean : bool
        Make sure the wavelets are zero mean. Defaults to True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    power : np.ndarray, shape (n_channels, n_frequencies, n_times)
        Induced power. Squared amplitude of time-frequency coefficients.
    itc : np.ndarray, shape (n_channels, n_frequencies, n_times)
        Phase locking value. [OPTIONAL; returned when return_itc is True]
    times : np.ndarray, shape (n_times, )
         Time vector for convenience based on n_times, sfreq and decim

    """
    n_trials, n_channels, n_times = data[:, :, ::decim].shape
    print(f'Data is {n_trials} trials and {n_channels} channels')
    n_frequencies = len(frequencies)
    print(f'Multitaper time-frequency analysis for {n_frequencies} frequencies')

    # Precompute wavelets for given frequency range to save time
    Ws = _dpss_wavelet(sfreq, frequencies, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, zero_mean=zero_mean)
    n_taps = len(Ws)
    print(f'Using {n_taps} tapers')
    n_times_wavelets = Ws[0][0].shape[0]
    if n_times <= n_times_wavelets:
        warnings.warn("Time windows are as long or longer than the epoch. "
                      "Consider reducing n_cycles.")
    psd = np.zeros((n_channels, n_frequencies, n_times))
    if return_itc:
        itc = np.zeros((n_channels, n_frequencies, n_times))

    for m in range(n_taps):
        psd_itc = (_time_frequency(data[:, c, :], Ws[m], use_fft, decim)
                   for c in range(n_channels))
        for c, (psd_c, itc_c) in enumerate(psd_itc):
            psd[c, :, :] += psd_c
            if return_itc:
                itc[c, :, :] += itc_c
    psd /= n_taps
    if return_itc:
        itc /= n_taps
    times = np.arange(n_times) / np.float(sfreq)
    if return_itc:
        return psd, itc, times
    else:
        return psd, times
