#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from source.audio_handler import HINT_audio_handler
from source.tf_transforms import my_istft, my_stft
from source.algorithms import misi, omisi
from source.helpers import record_wav, display_results, folder_handler

__author__ = 'Paul Magron -- IRIT'
__docformat__ = 'reStructuredText'


def apply_separation_algos(mixture_stft, spectrograms_target, src_ref, audio_folder_path, estim_or_oracle, win_length=256, hop_length=128, max_iter=15, fs=16000, win_type='hann'):
    """A function that applies all separation algorithms for benchmarking spectrogram inversion.
    Args:
        mixture_stft: numpy.ndarray (nfreqs, nframes) - input mixture STFT
        spectrograms_target: numpy.ndarray (nfreqs, nframes, nrsc) - the target sources' magnitude spectrograms
        src_ref: numpy.ndarray (nsamples, nrsc) - reference sources for computing the SDR over iterations
        audio_folder_path: string - the folder where the sounds must be recorded
        estim_or_oracle: string - 'oracle' or 'estim'
        win_length: int - the window length
        hop_length: int - the hop size of the STFT
        max_iter: int - number of iterations
        fs: int - sample rate
        win_type: string - window type
    Returns:
        sdr_online: list (5) - the SDR for online techniques: AM, oMISI with K=0,1,2 (mixture's phase initialization)
        and oMISI with K=1 (sinus phase initialization)
        sdr_misi: list (max_iter) - score (SI-SDR in dB) of MISI over iterations
        error_misi: list (max_iter) - loss function (magnitude mismatch) of MISI over iterations
    """
    # Record the references sources
    record_wav(src_ref, audio_folder_path, estim_or_oracle, 'ref', fs)

    # Initialize an empty list with online techniaues' scores
    sdr_online = []

    # Amplitude mask (equivalent to oMISI with 0 iteration)
    estim_sources, sdr = \
        omisi(mixture_stft, spectrograms_target, win_length=win_length, max_iter=0, hop_length=hop_length,
                    src_ref=src_ref, init_method='mix', future_frames=0)
    record_wav(estim_sources, audio_folder_path, estim_or_oracle, 'am', fs)
    sdr_online.append(sdr)

    # MISI
    estim_sources, error_misi, sdr_misi = misi(mixture_stft, spectrograms_target,  win_length=win_length, \
                     hop_length=hop_length, src_ref=src_ref, max_iter=max_iter, win_type=win_type)
    record_wav(estim_sources, audio_folder_path, estim_or_oracle, 'misi', fs)
    
    # Online MISI (for K=0,1,2 future frames)
    for K in range(3):
        # Run the algorithm
        estim_sources, sdr = \
            omisi(mixture_stft, spectrograms_target, win_length=win_length, max_iter=max_iter//(K+1), hop_length=hop_length,
                        src_ref=src_ref, init_method='mix', future_frames=K)
        # Record wav files
        record_wav(estim_sources, audio_folder_path, estim_or_oracle, 'omisi' + str(K), fs)
        # Store the SI-SDR
        sdr_online.append(sdr)

    # Sinus initialization (omisi with K=1)
    estim_sources, sdr = \
        omisi(mixture_stft, spectrograms_target, win_length=win_length, max_iter=max_iter//2, hop_length=hop_length,
                    src_ref=src_ref, init_method='sinus', future_frames=1)
    record_wav(estim_sources, audio_folder_path, estim_or_oracle, 'omisi_sin', fs)
    sdr_online.append(sdr)
    
    return sdr_online, sdr_misi, error_misi


def main(folders, parameters):
    """The main function that benchmarks all algorithm over the dataset
    Args:
        folders: dict with fields:
            'data': the dataset path
            'speakers': the name of the folder corresponding to the chosen speaker pair
            'outputs': the folder where the outputs (audio files, metrics, models) are stored
        parameters: dict with audio parameters:
            'sample_rate': int - the sample rate
            'win_length': int - the window length for the STFT
            'hop_length': int - the hop size of the STFT
            'n_fft': int - number of FFT points
            'fs': int - sample rate
            'win_type': string - window type
    """
    # Get parameters
    n_fft = parameters['n_fft']
    hop_length = parameters['hop_length']
    win_length = parameters['win_length']
    fs = parameters['sample_rate']
    win_type = parameters['win_type']
    
    # Create an object for each speaker to get the file list. Test mixtures are created by summing files from the lists
    test_data_1 = HINT_audio_handler(folders['data'], [folders['speakers'][0]], 'test', parameters['sample_rate'])
    test_data_2 = HINT_audio_handler(folders['data'], [folders['speakers'][1]], 'test', parameters['sample_rate'])
    test_data_1.get_files_list()
    test_data_2.get_files_list()
    
    # Number of sentences per speaker
    list_range = 2
    
    # Pre-allocate score
    metric_omisi = np.zeros([5, list_range ** 2])
    metric_misi = np.zeros([15, 2, list_range ** 2])

    # Loop over testing dataset
    ic = 0
    for file_num_1 in np.arange(list_range):
        for file_num_2 in np.arange(list_range, list_range * 2):
            # Load the test data for each sentence
            audio_in_1, audio_name_1 = test_data_1.get_file_from_list(file_num_1)
            audio_in_2, audio_name_2 = test_data_2.get_file_from_list(file_num_2)

            # Adjust to the same length and stack in an array
            min_len = min(len(audio_in_1), len(audio_in_2))
            audio_in_1 = audio_in_1[:min_len]
            audio_in_2 = audio_in_2[:min_len]
            src_ref = np.stack((audio_in_1, audio_in_2), axis=1)

            # STFTs
            src_ref_stft = my_stft(src_ref, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            mixture_stft = np.sum(src_ref_stft, axis=2)
            spectrograms_true = np.abs(src_ref_stft)

            # iSTFT (for having proper time domain size)
            src_ref = my_istft(src_ref_stft, hop_length=hop_length, win_length=win_length)

            # Create the folder to record audio files
            audio_1 = audio_name_1[audio_name_1.find('L'):][:-4]
            audio_2 = audio_name_2[audio_name_2.find('L'):][:-4]
            audio_folder_path = os.path.join(folders['outputs'], 'audio_files', audio_1 + '_' + audio_2)
            if not os.path.isdir(audio_folder_path):
                os.makedirs(audio_folder_path)
        
            # Separation algorithms
            sdr_online, sdr_misi, error_misi =\
                apply_separation_algos(mixture_stft, spectrograms_true, src_ref, audio_folder_path, 'oracle', \
                win_length=win_length, hop_length=hop_length, max_iter=15, fs=fs, win_type=win_type)
            
            # Record score
            metric_omisi[:, ic] = sdr_online
            metric_misi[:, 0, ic] = sdr_misi
            metric_misi[:, 1, ic] = error_misi

            ic += 1

    np.savez(folders['outputs'] + '/metrics.npz', metric_omisi=metric_omisi, metric_misi=metric_misi)


if __name__ == '__main__':
    # Pseudo-random numbers seed for reproducibility
    np.random.seed(117)
    store_seed = np.random.get_state()

    # Audio parameters
    parameters = {'sample_rate': 16000,
                  'win_length': 256,
                  'hop_length': 128,
                  'n_fft': 512,
                  'win_type': 'hann'
                  }
                  
    # Chosen speaker pair
    indx_speaker = 0             
    # Get folders for inputs / outputs
    folders = folder_handler(parameters, indx_speaker)
    
    # Run the evaluation
    main(folders, parameters)
    
    # Plot the results
    display_results(folders['outputs'])


# EOF
