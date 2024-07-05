import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os

import tensorflow as tf
"""
#Load segment audio classification model

model_path = r"best_model/"
model_name = "audio_NN3_grouping2019_10_01_11_40_45_acc_91.28"

# Model reconstruction from JSON file
# with open(model_path + model_name + '.json', 'r') as f:
#     model = model_from_json(f.read())

# # Load weights into the new model
# model.load_weights(model_path + model_name + '.h5')

# Replicate label encoder
lb = LabelEncoder()
lb.fit_transform(['Calling', 'Clapping', 'Falling', 'Sweeping', 'WashingHand', 'WatchingTV','enteringExiting','other'])
# array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int64)
#Some Utils
"""



model = tf.keras.models.load_model('new_model.h5')
lb = LabelEncoder()
specificActivities =  [ 'Clapping',"WashingHand",'WaterPouring','enteringExiting','others']

# specificActivities = [ 'Clapping', 'Falling','WatchingTV','enteringExiting','WaterPouring','SitStill','others']
# specificActivities = [ 'Clapping', 'Falling','WatchingTV','Walking','enteringExiting','others']
lb.fit_transform(specificActivities)

# Plot audio with zoomed in y axis
def plotAudio(output,time_axis = False,sr = 22050):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10))
    
    
    if time_axis == False:
        ax.set_xlim((0, len(output)))
        plt.plot(output, color='blue')
    else:
        ax.set_xlim((0, len(output)/sr))
        x_values = np.arange(0,len(output)/sr,1/sr)
        plt.plot(x_values,output,color='blue')
    ax.margins(2, -0.1)
    plt.show()

# Plot audio
def plotAudio2(output):
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))
    plt.plot(output, color='blue')
    ax.set_xlim((0, len(output)))
    plt.show()

# Split a given long audio file on silent parts.
# Accepts audio numpy array audio_data, window length w and hop length h, threshold_level, tolerence
# threshold_level: Silence threshold
# Higher tolence to prevent small silence parts from splitting the audio.
# Returns array containing arrays of [start, end] points of resulting audio clips
def split_audio(audio_data, w, h, threshold_level, tolerence=10):
    split_map = []
    start = 0
    data = np.abs(audio_data)
    threshold = threshold_level*np.mean(data[:25000])
    inside_sound = False
    near = 0
    for i in range(0,len(data)-w, h):
        win_mean = np.mean(data[i:i+w])
        if(win_mean>threshold and not(inside_sound)):
            inside_sound = True
            start = i
        if(win_mean<=threshold and inside_sound and near>tolerence):
            inside_sound = False
            near = 0
            split_map.append([start, i])
        if(inside_sound and win_mean<=threshold):
            near += 1
    return split_map

def minMaxNormalize(arr):
    mn = np.min(arr)
    mx = np.max(arr)
    return (arr-mn)/(mx-mn)

def predictSound(X):
    stfts = np.abs(librosa.stft(X, n_fft=512, hop_length=256, win_length=512))
    stfts = np.mean(stfts,axis=1)
    stfts = minMaxNormalize(stfts)
    result = model.predict(np.array([stfts]))
    predictions = [np.argmax(y) for y in result]
    return lb.inverse_transform([predictions[0]])[0]

    

# place concatenating audio files in a folder




""" VER 1"""
# import os  
  
# cur_dir = os.getcwd()
# folder = r"sound_clips/"
# folder_path = cur_dir + "/Dataset_audio/test/"  + 'sound_clips/'
# raw_audio, sr = librosa.load(folder_path + os.listdir(folder_path)[0]) # length of sound clip = len(raw_audio)/sr,sr = 22050 

# raw_audio, sr = librosa.load("C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\OK2.wav")

# # for file in os.listdir(folder_path):
# #     data, rate = librosa.load(folder_path + file)
# #     raw_audio = np.concatenate((raw_audio,data))
# noisy_part = raw_audio[0:50000]  # Empherically selected noisy_part position for every sample
# nr_audio = nr.reduce_noise(raw_audio,sr,False, noisy_part)
# plotAudio(nr_audio)
# plotAudio(nr_audio,time_axis = True)


# split_audio = split_audio(nr_audio, 512, 256, 1, tolerence=10)

# predict_audio([nr_audio[split_audio[0][0]:(split_audio[0][1]+1)]])





""" VER 2"""
# def predict_audio(data):
  
#     #trimming
#     trimmed, index = librosa.effects.trim(data, top_db=20, frame_length=512, hop_length=64)

#     # extract features
#     stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    
#     data = stft.T  # data = np.load(stft_save_path).T
   
#     data = np.mean(data,axis=0)
    
    
#     result = model.predict(data.reshape(-1,257))
#     predictions = [np.argmax(y) for y in result]
#     predicted_class = list(lb.classes_)[predictions[0]]
#     accuracy = result[0][predictions[0]]*100
#     print(list(lb.classes_))
#     print( predicted_class,accuracy ,'%')
#     print(result)
    
    
# # read audio data    
# audio_data, sample_rate = librosa.load("C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\OK2.wav")

# # noise reduction
# noisy_part = audio_data[0:25000]  
# reduced_noise = nr.reduce_noise(audio_data,sample_rate,False,noisy_part)

# split_audio_data = split_audio(reduced_noise, 512, 256, 1, tolerence=10)

# for i in range(len(split_audio)):
#     data = reduced_noise[split_audio_data[i][0]:split_audio[i][1]]
#     predict_audio(data)
    
    
""" VER 3"""
def predict_audio_continuously(file):
    audio_data, sample_rate = librosa.load(file)
    
    # INCREASE VOLUME
    factor = 20
    audio_data *= factor
    # noise reduction
    noisy_part = audio_data[0:25000]  
    reduced_noise = nr.reduce_noise(audio_data,sample_rate,False,noisy_part)

    
    split_audio_data = split_audio(reduced_noise, 512, 256, 1, tolerence=10)
    for i in range(len(split_audio_data)):
        data = reduced_noise[split_audio_data[i][0]:split_audio_data[i][1]]
        #trimming
        trimmed, index = librosa.effects.trim(data, top_db=20, frame_length=512, hop_length=64)
    
        # extract features
        stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
        
        data = stft.T  # data = np.load(stft_save_path).T
       
        data = np.mean(data,axis=0)
        
        
        result = model.predict(data.reshape(-1,257))
        predictions = [np.argmax(y) for y in result]
        predicted_class = list(lb.classes_)[predictions[0]]
        accuracy = result[0][predictions[0]]*100
        print(list(lb.classes_))
        print( predicted_class,accuracy ,'%')
        print(result)
    
# predict_audio_continuously("C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\OK2.wav")
predict_audio_continuously("C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\VideoAudioSave.wav")
# # read audio data    
# audio_data, sample_rate = librosa.load("C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\OK2.wav")

# # noise reduction
# noisy_part = audio_data[0:25000]  
# reduced_noise = nr.reduce_noise(audio_data,sample_rate,False,noisy_part)

# split_audio_data = split_audio(reduced_noise, 512, 256, 1, tolerence=10)

# for i in range(len(split_audio)):
#     data = reduced_noise[split_audio_data[i][0]:split_audio[i][1]]
#     predict_audio(data)
    