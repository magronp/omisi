#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from source.tf_transforms import my_stft, my_istft, norm_synthesis_window
from source.metrics import get_separation_score
import scipy
import librosa
from scipy import signal

__author__ = 'Paul Magron -- IRIT'
__docformat__ = 'reStructuredText'


def misi(mixture_stft, spectrograms_target, win_length, hop_length=None, src_ref=None, max_iter=15, win_type='hann'):
    """The multiple input spectrogram inversion algorithm for source separation.
    Args:
        mixture_stft: numpy.ndarray (nfreqs, nframes) - input mixture STFT
        spectrograms_target: numpy.ndarray (nfreqs, nframes, nrsc) - the target sources' magnitude spectrograms
        win_length: int - the window length
        hop_length: int - the hop size of the STFT
        src_ref: numpy.ndarray (nsamples, nrsc) - reference sources for computing the SDR over iterations
        max_iter: int - number of iterations
        win_type: string - window type
    Returns:
        estimated_sources: numpy.ndarray (nsamples, nrsc) - the time-domain estimated sources
        error: list (max_iter) - loss function (magnitude mismatch) over iterations
        sdr: list (max_iter) - score (SI-SDR in dB) over iterations
    """
    if hop_length is None:
        hop_length = win_length // 2

    compute_sdr = True
    if src_ref is None:
        compute_sdr = False

    # Parameters
    number_sources = spectrograms_target.shape[2]
    n_fft = (spectrograms_target.shape[0]-1)*2

    # Pre allocate SDR and error
    sdr = []
    error = []

    # Initialize the time domain estimates with the mixture
    mixture_time = my_istft(mixture_stft, hop_length=hop_length, win_length=win_length, win_type=win_type)
    estimated_sources = np.repeat(mixture_time[:, np.newaxis], number_sources, axis=1) / number_sources

    for iteration_number in range(max_iter):
        # STFT
        stft_reest = my_stft(estimated_sources, n_fft=n_fft, hop_length=hop_length, win_length=win_length, win_type=win_type)
        current_magnitude = np.abs(stft_reest)
        # Compute and distribute the mixing error
        mixing_error = mixture_stft - np.sum(stft_reest, axis=2)
        corrected_stft = stft_reest + np.repeat(mixing_error[:, :, np.newaxis], number_sources, axis=2) / number_sources
        # Normalize to the target amplitude
        stft_estim = corrected_stft * spectrograms_target / (np.abs(corrected_stft) + sys.float_info.epsilon)
        # Inverse STFT
        estimated_sources = my_istft(stft_estim, win_length=win_length, hop_length=hop_length, win_type=win_type)
        # BSS score
        if compute_sdr:
            sdr.append(get_separation_score(src_ref, estimated_sources))
        # Error
        error.append(np.linalg.norm(current_magnitude - spectrograms_target))

    return estimated_sources, error, sdr


def omisi(mixture_stft, spectrograms_target, win_length, hop_length=None, init_method='mix', future_frames=1, src_ref=None, phase_true=None, max_iter=5, win_type='hann'):
    """The online multiple input spectrogram inversion algorithm for source separation.
    Args:
        mixture_stft: numpy.ndarray (nfreqs, nframes) - input mixture STFT
        spectrograms_target: numpy.ndarray (nfreqs, nframes, nrsc) - the target sources' magnitude spectrograms
        win_length: int - the window length
        hop_length: int - the hop size of the STFT
        init_method: string - phase initialization method ('mix', 'sinus', or 'true')
        future_frames: int - number of future frames to account for
        src_ref: numpy.ndarray (nsamples, nrsc) - reference sources for computing the SDR over iterations
        phase_true: numpy.ndarray (nfreqs, nframes, nrsc) - the ground truth sources' phase (for ideal phase mask)
        max_iter: int - number of iterations
        win_type: string - window type
    Returns:
        estimated_sources: numpy.ndarray (nsamples, nrsc) - the time-domain estimated sources
        sdr: list (max_iter) - score (SI-SDR in dB) over iterations
    """
    # For a null look-ahead, use a slightly faster version of oMISI
    if future_frames == 0:
        estimated_sources, sdr = omisi_fast(mixture_stft, spectrograms_target, win_length, hop_length, init_method, src_ref, phase_true, max_iter, win_type)
        return estimated_sources, sdr

    if hop_length is None:
        hop_length = win_length // 2

    # Parameters
    n_freqs, n_frames, nsrc = spectrograms_target.shape
    n_fft = (n_freqs - 1) * 2

    # Pre allocate proper size time domain signals
    expected_signal_len = win_length + hop_length * (n_frames - 1)
    estimated_sources = np.zeros((expected_signal_len, nsrc))
    s_current = np.zeros((win_length + future_frames * hop_length, nsrc))

    # Initial phase allocation
    phase_current = np.repeat(np.angle(mixture_stft[:, 0:future_frames])[:, :, np.newaxis], nsrc, axis=2)

    # Loop over time frames
    for i in range(n_frames-future_frames):

        sample_start = i * hop_length
        mag = spectrograms_target[:, i:i+future_frames+1, :]

        # Initialization of the new frame
        if init_method == 'mix':
            phase_new = np.repeat(np.angle(mixture_stft[:, i+future_frames])[:, np.newaxis, np.newaxis], nsrc, axis=2)
        elif init_method == 'sinus':
            phase_new = phase_current[:, -1, :] + get_normalized_frequencies_multi_sources(mag[:, -1, :]) * 2 * np.pi * hop_length
            phase_new = phase_new.reshape((n_freqs, 1, nsrc))
        elif init_method == 'true':
            phase_new = phase_true[:, i+future_frames, :]
        else:
            raise ValueError('Unknown initialization scheme')

        phase_ini = np.concatenate((phase_current, phase_new), axis=1)
        Y_dft_corrected = mag * np.exp(1j * phase_ini)

        # partial iSTFT
        s_wind = my_istft(Y_dft_corrected, win_length=win_length, hop_length=hop_length, win_type=win_type)

        # Overlap add
        s_ola = s_current + s_wind

        for iter in range(max_iter):
            # partial STFT
            Y_dft = my_stft(s_ola, win_length=win_length, hop_length=hop_length, n_fft=n_fft, win_type='hanning')
            # Compute and distribute the mixing error
            mixing_error = mixture_stft[:, i:i+future_frames+1] - np.sum(Y_dft, axis=2)
            Y_dft_corrected = Y_dft + np.repeat(mixing_error[:, :, np.newaxis], nsrc, axis=2) / nsrc
            # Normalize to the target magnitude (GL)
            Y_dft_norm = mag * np.exp(1j * np.angle(Y_dft_corrected))
            # partial iSTFT
            s_wind = my_istft(Y_dft_norm, win_length=win_length, hop_length=hop_length, win_type=win_type)
            # Overlap add with the previous exited frame
            s_ola = s_current + s_wind

        phase_current = np.angle(Y_dft_corrected)[:, 1:, :]
        #estimated_sources[sample_start:(sample_start + hop_length), :] = s_ola[:hop_length, :]
        #s_current = np.concatenate((s_ola[hop_length:, :], np.zeros((hop_length, nsrc))))
        estimated_sources[sample_start:(sample_start + win_length + future_frames * hop_length), :] = s_ola
        
        # Update the current fixed segment (ignore the future frames but account for the overlapped past frames)
        s_partial = my_istft(np.reshape(Y_dft_norm[:, 0, :], (n_freqs, 1, nsrc)), win_length=win_length, hop_length=hop_length, win_type=win_type)
        s_current = np.concatenate((s_current[hop_length:win_length, :] + s_partial[hop_length:, :], np.zeros(((future_frames+1) * hop_length, nsrc))))
    
    sdr = []
    if not(src_ref is None):
        sdr = get_separation_score(src_ref, estimated_sources)

    return estimated_sources, sdr


def omisi_fast(mixture_stft, spectrograms_target, win_length, hop_length=None, init_method='mix', src_ref=None, phase_true=None, max_iter=5, win_type='hann'):
    """The online multiple input spectrogram inversion algorithm for source separation. This version is design to be faster when there is 0 future frame
    Args:
        mixture_stft: numpy.ndarray (nfreqs, nframes) - input mixture STFT
        spectrograms_target: numpy.ndarray (nfreqs, nframes, nrsc) - the target sources' magnitude spectrograms
        win_length: int - the window length
        hop_length: int - the hop size of the STFT
        init_method: string - phase initialization method ('mix', 'sinus', or 'true')
        src_ref: numpy.ndarray (nsamples, nrsc) - reference sources for computing the SDR over iterations
        phase_true: numpy.ndarray (nfreqs, nframes, nrsc) - the ground truth sources' phase (for ideal phase mask)
        max_iter: int - number of iterations
        win_type: string - window type
    Returns:
        estimated_sources: numpy.ndarray (nsamples, nrsc) - the time-domain estimated sources
        sdr: list (max_iter) - score (SI-SDR in dB) over iterations
    """
    if hop_length is None:
        hop_length = win_length // 2

    # Parameters
    n_freqs, n_frames, nsrc = spectrograms_target.shape
    n_fft = (n_freqs - 1) * 2

    # Define windows
    fft_window = librosa.filters.get_window(win_type, win_length, fftbins=True)
    ifft_window = norm_synthesis_window(fft_window, hop_length)
    fft_window = fft_window.reshape((-1, 1))
    fft_window_multi = np.repeat(fft_window, nsrc, axis=1)
    ifft_window = ifft_window.reshape((-1, 1))
    ifft_window_multi = np.repeat(ifft_window, nsrc, axis=1)

    # Pre allocate proper size time domain signals
    expected_signal_len = win_length + hop_length * (n_frames - 1)
    estimated_sources = np.zeros((expected_signal_len, nsrc))
    s_current = np.zeros((win_length, nsrc))

    # Initial phase allocation
    phase_current = np.repeat(np.angle(mixture_stft[:, 0])[:, np.newaxis], nsrc, axis=1)

    # Loop over time frames
    for i in range(n_frames):

        sample_start = i * hop_length
        mag = spectrograms_target[:, i, :]

        # Initialization using the mixture's phase
        if init_method == 'mix':
            phase_ini = np.repeat(np.angle(mixture_stft[:, i])[:, np.newaxis], nsrc, axis=1)
        elif init_method == 'sinus':
            phase_ini = phase_current + get_normalized_frequencies_multi_sources(mag) * 2 * np.pi * hop_length
        elif init_method == 'true':
            phase_ini = phase_true[:, i, :]
        else:
            raise ValueError('Unknown initialization scheme')
        Y_dft_corrected = mag * np.exp(1j * phase_ini)

        # iDFT
        Y_dft_norm = np.concatenate((Y_dft_corrected, Y_dft_corrected[-2:0:-1, :].conj()))
        ifft_buffer = scipy.ifft(Y_dft_norm, n_fft, 0).real
        s_wind = ifft_window_multi * ifft_buffer[:win_length, :]
        # Overlap add
        s_ola = s_current + s_wind

        for iter in range(max_iter):
            # windowing the current time segment
            y_win = fft_window_multi * s_ola
            # FFT
            Y_dft = scipy.fft(y_win, n_fft, 0)[:n_freqs, :]
            # Compute and distribute the mixing error
            mixing_error = mixture_stft[:, i] - np.sum(Y_dft, axis=1)
            Y_dft_corrected = Y_dft + np.repeat(mixing_error[:, np.newaxis], nsrc, axis=1) / nsrc
            # Normalize to the target magnitude (GL)
            Y_dft_norm = mag * np.exp(1j * np.angle(Y_dft_corrected))
            # inverse DFT
            Y_dft_norm = np.concatenate((Y_dft_norm, Y_dft_norm[-2:0:-1, :].conj()))
            ifft_buffer = scipy.ifft(Y_dft_norm, n_fft, 0).real
            # Overlap add
            s_wind = ifft_window_multi * ifft_buffer[:win_length, :]
            s_ola = s_current + s_wind

        phase_current = np.angle(Y_dft_corrected)
        estimated_sources[sample_start:(sample_start + hop_length), :] = s_ola[:hop_length, :]
        s_current = np.concatenate((s_ola[hop_length:, :], np.zeros((hop_length, nsrc))))

    sdr = []
    if not(src_ref is None):
        sdr = get_separation_score(src_ref, estimated_sources)

    return estimated_sources, sdr
    

def get_normalized_frequencies_multi_sources(mag, min_height=None):

    num_channels, nsrc = mag.shape
    freq_normalized = np.zeros_like(mag)

    for ind_source in range(nsrc):
        freq_normalized[:, ind_source] = get_normalized_frequencies(mag[:, ind_source], min_height)

    return freq_normalized


def get_normalized_frequencies(v, min_height=None):
    """A function that returns the normalized frequencies in a spectrum computed by QIFFT.
    Used for applying the phase model based on mixtures of sinusoids.
    Args:
        v: numpy.ndarray (nfreqs,) - input nonnegative spectrum
        min_height: int - minimum value above which tracking the peaks
    Returns:
        freq_normalized: numpy.ndarray (nfreqs,) - array containing the normalized frequencies in every channels
    """
    if min_height is None:
        min_height = max(v) * 0.01

    # Initialization
    num_channels = len(v)
    freq_normalized = np.linspace(0, 0.5, num_channels)
    boundary_up = 1

    # Detect the peaks in the spectra
    spectrum_peaks = signal.find_peaks(v, height=min_height, distance=2)[0]
    spectrum_peaks = list(spectrum_peaks)
    number_peaks = len(spectrum_peaks)

    # Loop over peaks
    if not (number_peaks == 0):
        for ind_peak in range(number_peaks):
            peak_channel = spectrum_peaks[ind_peak]
            # Get the neighboring bins values
            peak_neighbor = np.log(v[peak_channel - 1:peak_channel + 2])
            # QIFFT
            p = 0.5 * (peak_neighbor[0] - peak_neighbor[2]) / (
                        peak_neighbor[0] - 2 * peak_neighbor[1] + peak_neighbor[2])
            # Refined frequency and normalize (divide by n_fft)
            freq_normalized_current = (peak_channel + p) / ((num_channels - 1) * 2)
            # Update the boudaries (regions of influence)
            boundary_low = boundary_up
            if not(ind_peak==number_peaks-1):
                boundary_up = (peak_channel + spectrum_peaks[ind_peak + 1]) // 2
            else:
                boundary_up = num_channels
            # Fill the vector with normalized frequencies
            freq_normalized[boundary_low:boundary_up] = freq_normalized_current

    return freq_normalized


# EOF
