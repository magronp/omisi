#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import soundfile as sf
import glob
import librosa
import os

__author__ = 'Paul Magron -- IRIT; Gaurav Naithani -- TUNI'
__docformat__ = 'reStructuredText'


class HINT_audio_handler(object):

    def __init__(self, database_folder, speaker_folders=None, stage=None, fs=None, normalise=False):
        self.database_folder=database_folder
        self.speaker_folders=speaker_folders
        if stage == 'train':
            self.lists = ['L10', 'L11', 'L12', 'L13']
        elif stage == 'val':
            self.lists = ['L8', 'L9']
        else:
            self.lists = ['L1', 'L2']
        self.fs=fs
        self.normalise=normalise

    def get_file_from_list(self, file_number):
        """
        Reads the audio file at [FILE_NUMBER] in the list of audio files into memory and returns an array with the
        contents of that file. Resamples audio file to the fs of the AUDIO_HANDLER object. Normalises file to peak = 1
        if normalise flag is set for the audio handler object.

        :param file_number: The location in the list of files in SELF.SPEAKER_FILES that we wish to return.
        :return: Array with audio data of the desired file, normalised and resampled according to parameters.
        """

        # IPython.embed()
        file = self.speaker_files[file_number]
        print('Getting File  ', file)
        audio_data, samplerate = sf.read(file)

        audio_filename = os.path.basename(sf.info(file).name)
        audio_data_nonzero = audio_data[audio_data != 0]
        if self.normalise:
            audio_data_nonzero = audio_data_nonzero / np.max(np.abs(audio_data_nonzero))
        if samplerate != self.fs:
            audio_data_nonzero = librosa.core.resample(audio_data_nonzero, orig_sr=samplerate, target_sr=self.fs)
        return audio_data_nonzero, audio_filename

    def get_files_list(self):   #returns the list of files associated with a particular source
        print("Getting files for Folders" +str(self.speaker_folders) + "and lists " + str(self.lists))
        files_speaker = []
        for speaker in self.speaker_folders:
            for list in self.lists:
                string = self.database_folder + '/' + speaker + '/HINT_DK_' + speaker + '_' + list + '_*.wav'
                files_speaker = files_speaker + glob.glob(string)
        self.speaker_files=files_speaker
        self.number_of_files=len(files_speaker)
        #embed()
        return self.speaker_files

    def get_all_audio_from_list(self):     #loads all the audio listed, resampled and concatenated.
        """
        Loads the audio files referenced in the list at SELF.SPEAKER_FILES into memory.
        Converts the samplerate to that specified when the HINT_AUDIO_HANDLER object was created, and also normalises
        each file (to peak value=1) if that flag is set in the object.

        N.B. Normalisation is not currently recommended.
        :return: Returns an array with all the audio files concatenated into one long array.

        """
        #todo: check that a file list has been created
        print("Getting the audio for the files in list")

        files_list=self.speaker_files
        fs=self.fs
        returned_audio_data = np.array([])
        print(files_list)
        # First make all the training data for speaker 1
        #embed()
        audio_data = None
        for file in files_list:
            audio_data, samplerate = sf.read(file)
            audio_data_nonzero = audio_data[audio_data != 0]  # REMOVE ZERO ENTRIES
            if self.normalise:
                audio_data_nonzero=audio_data_nonzero/np.max(np.abs(audio_data_nonzero))
            if samplerate != fs:
                audio_data_nonzero = librosa.core.resample(audio_data_nonzero, orig_sr=samplerate, target_sr=fs)
            returned_audio_data = np.append(returned_audio_data, audio_data_nonzero)
            audio_data=returned_audio_data

        print("Done! Audio obtained")
        return audio_data

    def load_all_audio_from_file_list(self):
        """
        A simpler interface to call both of the get files list and get_all_audio_from_list methods.
        First gets the file list, then loads each file from that list and returns the audio concatenated.

        :return:  Returns an array with all the audio files concatenated into one long array.
        """

        self.get_files_list()                       # get the list of files
        audio_data=self.get_all_audio_from_list()   # load the files.
        return audio_data                           # Return the files

# EOF
