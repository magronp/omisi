#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import librosa
import matplotlib.pyplot as plt

__author__ = 'Paul Magron -- IRIT'
__docformat__ = 'reStructuredText'


def record_wav(sources, audio_folder_path, estim_or_oracle, algo_name, fs=16000):
    """A function that record the sources in the .wav format
    Args:
        sources: numpy.ndarray (nsamples, nsrc) - input time signals
        audio_folder_path: string - the folder where the sounds must be recorded
        estim_or_oracle: string - 'oracle' or 'estim'
        algo_name: string - the name of the algorithm used for computing the sources
        fs: int - sample rate
    """
    # Define the file names
    if algo_name == 'ref':
        file1 = os.path.join(audio_folder_path, 'sig1.wav')
        file2 = os.path.join(audio_folder_path, 'sig2.wav')
        # Record the mixture
        librosa.output.write_wav(os.path.join(audio_folder_path, 'mix.wav'), np.sum(sources, axis=1), fs)
    else:
        file1 = os.path.join(audio_folder_path, 'sig1_' + estim_or_oracle + '_' + algo_name + '.wav')
        file2 = os.path.join(audio_folder_path, 'sig2_' + estim_or_oracle + '_' + algo_name + '.wav')

    # Record wav
    librosa.output.write_wav(file1, sources[:, 0], fs)
    librosa.output.write_wav(file2, sources[:, 1], fs)
    

def folder_handler(parameters, indx_speaker=0):
    """Returns a dictionary with folders paths
    Args:
        parameters: dict with audio parameters:
            'sample_rate': int - the sample rate
            'win_length': int - the window length for the STFT
        indx_speaker: int - the index of the speaker pair
    Returns:
        folders: dict with fields:
            'data': the dataset path
            'speakers': the name of the folder corresponding to the chosen speaker pair
            'outputs': the folder where the outputs (audio files, metrics, models) are stored
    """
    # Get the working directory
    working_folder = os.getcwd()
    
    # General info about the dataset
    dataset_name = 'HINT'
    speaker_pairs = ['F1_F2', 'M1_F1', 'M1_M2']
    database_folder = working_folder + '/data/HINT'

    # Pick the speaker pair and create a subfolder
    chosen_speaker_pair = speaker_pairs[indx_speaker]
    speaker_folders = [chosen_speaker_pair[:len(chosen_speaker_pair)//2], chosen_speaker_pair[len(chosen_speaker_pair)//2+1:]]

    # Create a subfolder for the corresponding window length
    win_length_folder = str(int(parameters['win_length'] / parameters['sample_rate'] * 1000)) + 'ms'
    
    # Folder to store outputs (model, training log, audio files and metrics)
    outputs_folder_path = os.path.join(working_folder, 'outputs/', dataset_name, chosen_speaker_pair, win_length_folder)
    if not os.path.isdir(outputs_folder_path):
        os.makedirs(outputs_folder_path)

    folders = {'data': database_folder,
          'speakers': speaker_folders,
          'outputs': outputs_folder_path
          }
              
    return folders


def display_results(outputs_folder_path):
    """Display inline the mean results for the separation and plot the error/SDR for MISI over iterations
    Args:
        outputs_folder_path: string - the folder where the 'metrics.npz' file is stored
    """
    # Load npz data
    data_loader = np.load(outputs_folder_path + '/metrics.npz')
    metric_omisi = data_loader['metric_omisi']
    metric_misi = data_loader['metric_misi']
    mean_metric_misi = np.mean(metric_misi, axis=2)

    # Display the mean results
    print(np.mean(metric_omisi, axis=1))
    print(mean_metric_misi[-1, 0])

    # Plot the loss and SI-SDR over iterations for MISI
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.log10(mean_metric_misi[:, 1]))
    plt.xlabel('Iterations')
    plt.title('log(Error)')
    plt.subplot(1, 2, 2)
    plt.plot(mean_metric_misi[:, 0])
    plt.xlabel('Iterations')
    plt.title('SI-SDRi (dB)')
    plt.tight_layout()
    plt.show()
    
    
# EOF
