#!/usr/bin/env python
# -*- coding: utf-8 -*-

import librosa
import numpy as np
import scipy

__author__ = 'Paul Magron -- IRIT'
__docformat__ = 'reStructuredText'


def my_stft(x, n_fft=2048, hop_length=None, win_length=None, win_type='hann', dtype=np.complex64):
    """Short-time Fourier transform
    Args:
        x: numpy.ndarray (nsamples, nsrc) - input time signals
        hop_length: int - hop size in samples
        win_length: int - window length in samples
        win_type: string - window type
    Returns:
        stft_matrix: numpy.ndarray (nfreqs, nframes, nsrc) - STFT matrix
    """
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    dims = x.shape
    len_sig = dims[0]
    if len(dims) == 1:
        nsrc = 1
        x = x[:, np.newaxis]
    else:
        nsrc = dims[1]

    # Define window and repeat it for multi sources
    fft_window = librosa.filters.get_window(win_type, win_length, fftbins=True)
    fft_window = fft_window.reshape((-1, 1))
    fft_window_multi = np.repeat(fft_window, nsrc, axis=1)

    # Pre-allocate the STFT matrix
    nframes = 1 + int((len_sig - win_length) / hop_length)
    nfreqs = int(1 + n_fft // 2)
    stft_matrix = np.empty((nfreqs, nframes, nsrc),
                           dtype=dtype,
                           order='F')

    # Loop over time frames - windowing and DFT
    for frame_index in range(nframes):
        time_beg = frame_index * hop_length
        fft_buffer = x[time_beg:time_beg + win_length, :] * fft_window_multi
        stft_matrix[:, frame_index, :] = scipy.fft(fft_buffer, n_fft, 0)[:nfreqs, :]

    # Remove the last dimension if there is only one source
    if len(dims)==1:
        stft_matrix = stft_matrix[:, :, 0]

    return stft_matrix


def my_istft(stft_matrix, hop_length=None, win_length=None, win_type='hann', dtype=np.float64):
    """inverse STFT
    Args:
        stft_matrix: numpy.ndarray (nfreqs, nframes, nsrc) - input STFT matrix
        hop_length: int - hop size in samples
        win_length: int - window length in samples
        win_type: string - window type
    Returns:
        y: numpy.ndarray (nsamples, nsrc) - synthesized time signals
    """
    dims = stft_matrix.shape
    nfreqs, nframes = dims[:2]
    n_fft = 2 * (nfreqs - 1)
    if len(dims) == 2:
        nsrc=1
        stft_matrix = stft_matrix[:, :, np.newaxis]
    else:
        nsrc=dims[2]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    # Define window and repeat it for multi sources
    ifft_window = librosa.filters.get_window(win_type, win_length, fftbins=True)
    ifft_window = norm_synthesis_window(ifft_window, hop_length)
    ifft_window = ifft_window.reshape((-1, 1))
    ifft_window_multi = np.repeat(ifft_window, nsrc, axis=1)

    # Initialize time domain signals
    expected_signal_len = win_length + hop_length * (nframes - 1)
    y = np.zeros((expected_signal_len, nsrc), dtype=dtype)

    # Loop over frames - iDFT and overlap-add
    for i in range(nframes):
        sample_start = i * hop_length
        spec = stft_matrix[:, i, :]
        spec = np.concatenate((spec, spec[-2:0:-1, :].conj()), 0)
        ifft_buffer = scipy.ifft(spec, n_fft, 0).real
        ytmp = ifft_window_multi * ifft_buffer[:win_length, :]
        y[sample_start:(sample_start + win_length), :] = y[sample_start:(sample_start + win_length), :] + ytmp

    # Remove the last dimension if there is only one source
    if len(dims) == 2:
        y = y[:, 0]

    return y

def norm_synthesis_window(wind, hop_length):
    """Computes a synthesis window for the iSTFT
    According to: Daniel W. Griffin and Jae S. Lim, `Signal estimation\
    from modified short-time Fourier transform,` IEEE Transactions on\
    Acoustics, Speech and Signal Processing, vol. 32, no. 2, pp. 236-243,\
    Apr 1984.
    Args:
        wind: numpy.ndarray (win_length,) - analysis window
        hop_length: int - hop size in samples
    Returns:
        syn_w: numpy.ndarray (win_length,) - synthesis window
    """
    window_size = len(wind)
    syn_w = wind
    syn_w_prod = syn_w ** 2.
    syn_w_prod.shape = (window_size, 1)
    redundancy = int(window_size / hop_length)
    env = np.zeros((window_size, 1))

    for k in range(-redundancy, redundancy + 1):
        env_ind = (hop_length * k)
        win_ind = np.arange(1, window_size + 1)
        env_ind += win_ind

        valid = np.where((env_ind > 0) & (env_ind <= window_size))
        env_ind = env_ind[valid] - 1
        win_ind = win_ind[valid] - 1
        env[env_ind] += syn_w_prod[win_ind]

    syn_w = syn_w / env[:, 0]

    return syn_w

# EOF
