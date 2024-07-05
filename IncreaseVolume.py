# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:05:32 2023

@author: mgt
"""

import librosa
import soundfile as sf
import os



folder_path = "C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\SaveSoundSampleContinuously"
folder_path_save = "C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\SaveSoundSampleContinuously_HighVolume"
print(os.listdir(folder_path))


for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path,file)
    save_file_path = os.path.join(folder_path_save,file)
    data, sr = librosa.load(file_path)
    factor = 20
    data *= factor
    
   
    sf.write(save_file_path, data, sr)