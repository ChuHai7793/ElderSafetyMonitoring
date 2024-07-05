# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:21:07 2023

@author: mgt
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:45:51 2023

@author: mgt
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:07:56 2023

@author: mgt
"""



from os import startfile

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from datetime import date,timedelta,datetime
import uuid
# from predict import *
from PIL import ImageTk, Image
import cv2
import keyboard
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
import pyaudio
import keyboard
import tensorflow as tf
import wave
import numpy as np
import time
import threading
from tkVideoPlayer import TkinterVideo
from tkvideoutils import VideoPlayer
from ttkwidgets import TickScale
from moviepy.editor import AudioFileClip
def save_STFT(file,save_file):
    # read audio data
    audio_data, sample_rate = librosa.load(file)
    # print(type(audio_data))
    
    # noise reduction
    noisy_part = audio_data[0:25000]  
    reduced_noise = nr.reduce_noise(audio_data,sample_rate,False,noisy_part)
    
    #trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    
    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    # save features
    np.save(save_file, stft)
    
    return stft




##################### LOAD MODEL #####################
def save_audio(pyAudioObj,saveAudioFile, frame,RATE):
        # Open and Set the data of the WAV file
        file = wave.open(saveAudioFile, 'wb')
        file.setnchannels(1)
        file.setsampwidth(pyAudioObj.get_sample_size(pyaudio.paFloat32))
        file.setframerate(RATE)
         
        #Write and Close the File
        file.writeframes(b''.join(frame))
        file.close()  
def predictSoundContinuously_fromMic(Model,Label,file):
 
    audio_data, sample_rate = librosa.load(file)
    # print(type(audio_data))
    
    # INCREASE VOLUME
    factor = 20
    audio_data *= factor
    
    
    # noise reduction
    noisy_part = audio_data[0:25000]  
    reduced_noise = nr.reduce_noise(audio_data,sample_rate,False,noisy_part)
    
    #trimming
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=20, frame_length=512, hop_length=64)
    
    # extract features
    stft = np.abs(librosa.stft(trimmed, n_fft=512, hop_length=256, win_length=512))
    
    stft = np.mean(stft,axis=1)
    result = Model.predict(stft.reshape(-1,257))
    predictions = [np.argmax(y) for y in result]
    print(list(Label.classes_)[predictions[0]], result[0][predictions[0]]*100 ,'%')
    
    return list(Label.classes_)[predictions[0]]
    
##################### LOAD MODEL #####################
model = tf.keras.models.load_model('models\\new_model.h5')
# model = tf.keras.models.load_model('my_model.h5')

lb = LabelEncoder()
specificActivities = [ 'Clapping',"WashingHand",'WaterPouring','enteringExiting','others']
# specificActivities =[ 'Clapping',"Falling", 'WatchingTV','enteringExiting','others']
# specificActivities = [ 'Clapping', 'Falling','WatchingTV','enteringExiting','WaterPouring','SitStill','others']
# specificActivities = [ 'Clapping', 'Falling','WatchingTV','Walking','enteringExiting','others']
lb.fit_transform(specificActivities)





## model for enterExiting video
demo_model1 = tf.keras.models.load_model('models\\demo_model1.h5')       
demo_specificActivities1 = [ 'Clapping',"Falling", 'WatchingTV','enteringExiting','others']
# demo_specificActivities = [ 'Clapping',"WashingHand",'WaterPouring','enteringExiting','others']

demo_lb1 = LabelEncoder()
demo_lb1.fit_transform(demo_specificActivities1)

## model for HandWashing video
demo_model2 = tf.keras.models.load_model('models\\demo_model2.h5')       
demo_specificActivities2 = [ 'Clapping',"WashingHand",'WaterPouring','enteringExiting','others']
# demo_specificActivities = [ 'Clapping',"WashingHand",'WaterPouring','enteringExiting','others']

demo_lb2 = LabelEncoder()
demo_lb2.fit_transform(demo_specificActivities2)

## model for enterExiting Continue Prediction
model1 = tf.keras.models.load_model('models\\modelEnterExit.h5')
# model = tf.keras.models.load_model('my_model.h5')

lb1 = LabelEncoder()

specificActivities1 =[ 'Clapping',"Falling", 'WatchingTV','enteringExiting','others']
# specificActivities = [ 'Clapping', 'Falling','WatchingTV','enteringExiting','WaterPouring','SitStill','others']
# specificActivities = [ 'Clapping', 'Falling','WatchingTV','Walking','enteringExiting','others']
lb1 = LabelEncoder()
lb1.fit_transform(specificActivities)




##################### INTERFACE #####################    
class AudioRecognitionApp(tk.Tk):
    def __init__(self):
#        super(GenerateKeyApp,self).__init__()
#        super().__init__(master)  
        super().__init__('AudioRecognitionApp')
        self.title('AudioRecognitionApp')        
        # self.geometry("900x800")  
        # self.resizable(width= 200, height=200)
        
        
        ############# ENTRY ########################
        self.EntryFrame = tk.Frame(self) 
        self.EntryFrame.pack(side = 'top',fill ='x' )

             
        self.prediction_var = tk.StringVar()
        self.accuracy_var = tk.StringVar()
        self.file = None
  
        
        tk.Label( self.EntryFrame,text = "Prediction").grid(row = 1, column = 0,sticky = 'w',columnspan=1, pady = 2)         
        self.prediction_text = tk.Entry( self.EntryFrame,textvariable = self.prediction_var, bd = 5,width= 40)
        self.prediction_text.grid(row = 1, column = 1,columnspan = 1,sticky = 'e', pady = 2) 
        
        tk.Label(  self.EntryFrame,text = "Accuracy").grid(row = 2, column = 0,sticky = 'w',columnspan=1, pady = 2)      
        self.Accuracy_text = tk.Entry( self.EntryFrame,textvariable = self.accuracy_var, bd = 5,width= 40)
        self.Accuracy_text.grid(row = 2, column = 1,columnspan = 1,sticky = 'e', pady = 2) 
        
        
        
        ############# BUTTON ########################
        self.BtnFrame = tk.Frame(self) 
        self.BtnFrame.pack(side = 'top',fill ='x' )
        
        self.browser_button = tk.Button( self.BtnFrame,text='ChooseAudioFile',height = 5,width = 15,
                                          command=self.fileDialog_audio ).grid(row = 3, column =0,columnspan=1,
                                                                               sticky = 'w', pady = 2)
        self.detect_button = tk.Button( self.BtnFrame,text='Detect',height = 5,width = 15,
                                          command= self.predict_audio ).grid(row = 3, column = 1,columnspan=1,
                                                                             sticky = 'w', pady = 2)
        self.chooseVideo_btn = tk.Button( self.BtnFrame,text='chooseVideo',height = 5,width = 15,
                                          command=self.choose_video ).grid(row = 3, column = 2,columnspan=1,
                                                                           sticky = 'e', pady = 2)
        self.SaveAudioFile_btn = tk.Button( self.BtnFrame,text='SaveAudio',height = 5,width = 15,
                                           command=lambda:self.SaveAudioFile(self.video_file) ).grid(row = 3, column = 3,columnspan=1,
                                                                         sticky = 'e', pady = 2)
         
        
                                                                                                     
        self.predict_video_btn = tk.Button( self.BtnFrame,text='predict video',height = 5,width = 15,
                                           command=lambda:self._thread_predict_audio_continuously(self.audio_path)).grid(row = 3, column = 4,columnspan=1,
                                                                         sticky = 'e', pady = 2)        
                                                                                                                     
        self.stream_video_btn = tk.Button( self.BtnFrame,text='stream video',height = 5,width = 15,
                                          command=self.video_stream ).grid(row = 3, column = 6,columnspan=1,
                                                                              sticky = 'e', pady = 2)
        self.close_video_btn = tk.Button( self.BtnFrame,text='close video',height = 5,width = 15,
                                          command=self.close_video ).grid(row = 3, column = 7,columnspan=1,
                                                                          sticky = 'e', pady = 2)
        self.continuous_predict_btn = tk.Button( self.BtnFrame,text='continuous predict',height = 5,width = 15,
                                          command=self.continuous_soundClassify_from_mic_thread ).grid(row = 3, column = 8,columnspan=1,
                                                                          sticky = 'e', pady = 2)
        self.stop_predict_btn = tk.Button( self.BtnFrame,text='stop predict',height = 5,width = 15,
                                  command=self.stop_predict ).grid(row = 3, column = 9,columnspan=1,
                                                                  sticky = 'e', pady = 2)                                                                                                       


        ############# VIDEO ########################                                                                                                                                         
        self.VideoFrame = tk.Frame(self) 
        self.VideoFrame.pack(side = 'left',fill ='both')                                                                   
        
        self.image_icon = Image.open("C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\Video-Icon-crop.png")
        self.image_icon = self.image_icon.resize((900,500), Image.LANCZOS)
        self.image_icontk = ImageTk.PhotoImage(self.image_icon)   
                                                      
        self.video = tk.Label( self.VideoFrame)
        self.video.grid(row = 0, column = 0,columnspan=1,sticky = 'w', pady = 2)
                                                
        self.video.imgtk = self.image_icontk
        self.video.configure(image=self.image_icontk)
        
        
        self.slider_var = tk.IntVar(self.VideoFrame)
        self.slider = TickScale(self.VideoFrame, orient="horizontal", variable=self.slider_var)
        self.slider.grid(row = 2, column = 0,columnspan=1,sticky = 'w', pady = 2)
        
        self.close_flag = 0
        self.stop_predict_flag = 0   
        self.stream_flag = 1
        self.stream_cap = cv2.VideoCapture(0)

        
        ############# Timer ######################## 
        self.TimerFrame = tk.Frame(self) 
        self.TimerFrame.pack(side = 'top',fill ='x')  
        self.Timer_Label = tk.Label( self.TimerFrame,text = "Timer",
                                    width = 5,height=2,relief="raise",
                                    font=("Arial", 25))
        self.Timer_Label.grid(row = 0, column = 1,sticky = 'we',columnspan=1, pady = 1)         
        
        ############# ALARM ########################  
        self.AlarmFrame = tk.Frame(self) 
        self.AlarmFrame.pack(side = 'top',fill ='x')  
        self.Alarm_Label = tk.Label( self.AlarmFrame,text = "Prediction",bg ="#A0A0A0",
                                    width =42, height=18,relief="raise",
                                    font=("Arial", 25))
        self.Alarm_Label.grid(row = 0, column = 1,sticky = 'e',columnspan=1, pady = 1)         
        
        self.audio_path = "C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\VideoAudioSave.wav"
    # def refresh(self):
    #     self.destroy()
    #     self.__init__()
        
    
    def __thread_timer_Display(self):
        x = threading.Thread(target=self.timer_Display, daemon=True)
        x.start()
               
        
    def timer_Display(self):
        self.timer_flag = 1
        self.counter = 0
        while (self.timer_flag == 1 and self.close_flag == 0):
            self.counter = self.counter + 1
            time.sleep(1)
            self.Timer_Label.configure(text =self.counter,bg ='#00FF00')  
        self.Timer_Label.configure(text ='Timer',bg ='#d9d9d9')     
    def SaveAudioFile(self,video_file):
       
        temp_audioclip = AudioFileClip(video_file)
        temp_audioclip.write_audiofile(self.audio_path, codec='pcm_s16le', verbose=False, logger=None)
        temp_audioclip.close()
        print("FINISH SAVE AUDIO")
           

    def __thread_display_videoTV(self):
        
        x = threading.Thread(target=self.display_videoTV,  args=(self.video_file,),daemon=True)
        x.start()  
    def __thread_display_videoDoor(self):
        time.sleep(1)
        x = threading.Thread(target=self.display_videoTV,  args=(self.video_file,),daemon=True)
        x.start()  
    def display_videoTV(self,video_file):
        startfile(self.video_file)
        # time.sleep(4)
        # # Create a VideoCapture object and read from input file
        # cap = cv2.VideoCapture(video_file)
        # cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        # # set your desired size
        # cv2.resizeWindow('Video', 900, 500)
        # # fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # # delay_time = float(1 /fps) - time.monotonic() % (float(1 / fps))  
        # # time.sleep(delay_time)
        # # Check if camera opened successfully
        # if (cap.isOpened()== False):
        #     print("Error opening video file")
        
        # # Read until video is completed
        # while(cap.isOpened()):
        #     # time.sleep(0.00000001) 
        # # Capture frame-by-frame
        #     ret, frame = cap.read()
        #     if ret == True:
        #     # Display the resulting frame
        #         cv2.imshow('Video', frame)
                  
        #     # Press Q on keyboard to exit
        #         if cv2.waitKey(25) & (0xFF == ord('q')or cv2.getWindowProperty('Video', 0) < 0):
        #             break
          
        # # Break the loop
        #     else:
        #         break
          
        # # When everything done, release
        # # the video capture object
        # cap.release()
          
        # Closes all the frames
        cv2.destroyAllWindows()    
        

    def __thread_display_video(self):
        x = threading.Thread(target=self.display_video,  args=(self.video_file,),daemon=True)
        x.start()  
    def display_video(self,video_file):
        
        # Create a VideoCapture object and read from input file
        cap = cv2.VideoCapture(video_file)
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        # set your desired size
        cv2.resizeWindow('Video', 900, 500)
        
        # Check if camera opened successfully
        if (cap.isOpened()== False):
            print("Error opening video file")
          
        # Read until video is completed
        while(cap.isOpened()):
              
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
            # Display the resulting frame
                cv2.imshow('Video', frame)
                  
            # Press Q on keyboard to exit
                if cv2.waitKey(25) & (0xFF == ord('q')or cv2.getWindowProperty('Video', 0) < 0):
                    break
          
        # Break the loop
            else:
                break
          
        # When everything done, release
        # the video capture object
        cap.release()
          
        # Closes all the frames
        cv2.destroyAllWindows()
        
    def fileDialog_audio(self):
        
        self.audio_file = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("all files","*.*"),("jpeg files","*.jpg")))    
        
    def close_video(self):
        self.close_flag = 1
                                                              
        self.video = tk.Label( self.VideoFrame)
        self.video.grid(row = 0, column = 0,columnspan=1,sticky = 'w', pady = 2)
                                                
        self.video.imgtk = self.image_icontk
        self.video.configure(image=self.image_icontk)

        self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')  
    
  
        
    def video_stream(self):
        # Capture from camera 
        
        _, frame = self.stream_cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)    
        img = Image.fromarray(cv2image)
        img= img.resize((600,500), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video.imgtk = imgtk
        self.video.configure(image=imgtk)
        if  keyboard.is_pressed("x") or self.close_flag == 1:
            self.close_flag = 0 
            # image_icon = Image.open("C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\Video-Icon-crop.png")
            # image_icontk = ImageTk.PhotoImage(image_icon)     
            self.video.imgtk = self.image_icontk
            self.video.configure(image=self.image_icontk)
            self.stream_cap.release()
            self.stream_cap = cv2.VideoCapture(0)
        else:
            self.video.after(1, self.video_stream)  

    def choose_video(self):   
        self.video_file = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("all files","*.*"),("jpeg files","*.jpg")))  
        self.video_cap = cv2.VideoCapture(self.video_file)   
        
    
    def showVideo_Thread(self):
        x = threading.Thread(target=self.showVideo, daemon=True)
        x.start()  

    def showVideo(self):
        
        # Capture from camera 
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        
        frames = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
 
        seconds = (frames / fps)
        print(seconds)
        
        
        success, frame = self.video_cap.read()
 
        if success: # CHECK IF RECEIVE ANY FRAME
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)    
            img = Image.fromarray(cv2image)
            img= img.resize((900,500), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video.imgtk = imgtk
            self.video.configure(image=imgtk)
            
            if  keyboard.is_pressed("x") or self.close_flag == 1:
                self.close_flag = 0 
                self.video.imgtk = self.image_icontk
                self.video.configure(image=self.image_icontk)
                self.video_cap.release()
                # self.video_cap = cv2.VideoCapture(0) 
            else:
                # self.video.after(int(1000/fps), self.showVideo)
                delay_time = float(1 /fps) - time.monotonic() % (float(1 / fps))      
                # delay_time = float(1 /fps)  
                time.sleep(delay_time)
                self.showVideo()
                
                # # self.video.after(delay_time, self.showVideo)
                
        else: # END WHEN NO MORE FRAME ACQUIRED
            self.close_flag = 0 
            self.video.imgtk = self.image_icontk
            self.video.configure(image=self.image_icontk)
            self.video_cap.release()
            # self.video_cap = cv2.VideoCapture(0)  
        
    def predict_audio(self):
      
        print(self.audio_file)
        stft_save_path = self.audio_file.split('.')[0]+'.npy'
        stft = save_STFT(self.audio_file,stft_save_path)
        
        data = stft.T  # data = np.load(stft_save_path).T
       
        data = np.mean(data,axis=0)
        
        
        result = model.predict(data.reshape(-1,257))
        predictions = [np.argmax(y) for y in result]
        predicted_class = list(lb.classes_)[predictions[0]]
        accuracy = result[0][predictions[0]]*100
        print(list(lb.classes_))
        print( predicted_class,accuracy ,'%')
        print(result)
        self.prediction_var.set(predicted_class)
        self.accuracy_var.set(accuracy)
        
    def continuous_soundClassify_from_mic_thread(self):
        x = threading.Thread(target=self.continuous_soundClassify_from_mic, daemon=True)
        x.start()     
        
        
    def continuous_soundClassify_from_mic(self):
        
        CHUNKSIZE = 22050 # fixed chunk size. Each chunk will consist of 22050 samples
        RATE = 22050 # Record at 22050 samples per second
        """
        Store data in chunks for 3 seconds
        for i in range(0, int(RATE / CHUNKSIZE * time_in_seconds)):
            data = stream.read(chunk)
            frames.append(data)
        """
        time_in_seconds = 3
        
    
        # initialize portaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, 
                        channels=1, 
                        rate=RATE, 
                        input=True, 
                        frames_per_buffer=CHUNKSIZE)
    
    
        i = 0
        j =1
        frames = []
    
        while(self.stop_predict_flag == 0):
            
            print(i)
            if keyboard.is_pressed("x") :
                print("You pressed x to exit")
                break
            
            # Read chunk and load it into numpy array.
            if i < int(RATE / CHUNKSIZE * time_in_seconds):
                data = stream.read(CHUNKSIZE)
                frames.append(data)
                i = i+1
            else:
                i = 0
                saveAudioFile  = "soundsample.wav"
                
                print(saveAudioFile)
                save_audio(p,saveAudioFile, frames,RATE)
                # save_audio(saveAudioFile, frames)
                self.predicted_result = predictSoundContinuously_fromMic( model,lb,"soundsample.wav")
                          
                if self.predicted_result in specificActivities and self.predicted_result!= 'others':
                    self.Alarm_Label.configure(text =self.predicted_result,bg ='red')
                    time.sleep(2)
                    self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')
                else:
                    self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')
                
                
                j = j+1
                frames = []
        self.stop_predict_flag = 0         
        # close stream
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("END")
        
        
    def stop_predict(self):
        self.stop_predict_flag = 1

    """
    SPLIT A LONG VIDEO TO SMALLER PERIOD AND MAKE PREDICTION
    """
    def _thread_predict_audio_continuously(self,file):
        
        y = threading.Thread(target=self.predict_audio_continuously,  args=(file,), daemon=True)
        y.start()     
        # self.player.play()

            
    def predict_audio_continuously(self,file_name):

        self.close_flag = 0
        print(self.video_file)
        print(type(self.video_cap))
        time_in_seconds = 5
        file = wave.open(file_name, 'rb')
        RATE = file.getframerate() # Record at 22050 samples per second
        CHUNKSIZE = 22050 # fixed chunk size. Each chunk will consist of 22050 samples
        
        # initialize portaudio
        p1 = pyaudio.PyAudio()
        stream1 = p1.open(format=p1.get_format_from_width(file.getsampwidth()), 
                        channels=file.getnchannels(), 
                        rate=file.getframerate(), 
                        output=True, 
                        frames_per_buffer=CHUNKSIZE)
    
    
        i = 0
        j =1
        frames = []
        data = file.readframes(CHUNKSIZE)
        # self.showVideo_Thread()
        # self.player.play()
        # self.__thread_display_video()
        if "demo" in self.video_file:
            if "door" in self.video_file:
                self.__thread_display_videoDoor()
                
                x = threading.Thread(target=self.__predictTV_Thread, daemon=True)
                x.start() 
            else:
                self.__thread_display_videoTV()
              
                x = threading.Thread(target=self.__predictTV_Thread, daemon=True)
                x.start() 
        else:
            self.__thread_display_videoTV()
              
      
        while(data != b'' and self.close_flag ==0):
            # print(self.close_flag)
            # print(i)
            if keyboard.is_pressed("x") :
                print("You pressed x to exit")
                break
            
            # Read chunk and load it into numpy array.
            if i < int(RATE / CHUNKSIZE * time_in_seconds):
                # Read data in chunks
                data = file.readframes(CHUNKSIZE)
                time.sleep(0.5)               # print(type(data))
                stream1.write(data)
                frames.append(data)
                i = i+1
            else:
                
                data = file.readframes(CHUNKSIZE)
                time.sleep(0.5)
                stream1.write(data)
                
                i = 0
                if "demo" not in self.video_file:
                    saveAudioFile  = "soundsample.wav"               
                    # print(saveAudioFile)
                    save_audio(p1,saveAudioFile, frames,RATE)
                    # Start predicting and show alarm
                    y = threading.Thread(target=self.__Thread_predictSoundFromVideo, daemon=True)
                    y.start() 
                
                j = j+1
                frames = []
             
        self.close_flag = 0
              
        # close stream
        stream1.stop_stream()
        stream1.close()
        p1.terminate()
        print("END")  

    def __Thread_predictSoundFromVideo(self):  
        if "door" in self.video_file:
            try:
                predicted_result = predictSoundContinuously_fromMic( demo_model1,demo_lb1,"soundsample.wav")
                          
                if predicted_result in demo_specificActivities1 and predicted_result!= 'others':         
                    self.Alarm_Label.configure(text = "EnterExiting",bg ='blue')
                    time.sleep(2)
                    self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')
                else:
                    self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')  
            except:pass
        elif "HandWash" in self.video_file:
            try:
                predicted_result = predictSoundContinuously_fromMic( demo_model2,demo_lb2,"soundsample.wav")
                          
                if predicted_result in demo_specificActivities2 and predicted_result!= 'others':         
                    if predicted_result == "Clapping":
                        time.sleep(2)
                        self.Alarm_Label.configure(text = "WashingHand",bg ='blue')
                    else:
                        self.Alarm_Label.configure(text = predicted_result,bg ='blue')
                        time.sleep(2)
                        self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')
                else:
                    self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')  
            except:pass
        else:
            try:
                predicted_result = predictSoundContinuously_fromMic( model,lb,"soundsample.wav")
                          
                if predicted_result in specificActivities and predicted_result!= 'others':
 
                    self.Alarm_Label.configure(text = predicted_result,bg ='red')
                    time.sleep(2)
                    self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')
                else:
                    self.Alarm_Label.configure(text ='Recording...',bg ='#00FF00')  
            except:pass   
            
    # def predictTV_Thread(self):
        
        
    #     x = threading.Thread(target=self.predictTV, daemon=True)
    #     x.start() 
    #     time.sleep(3)
    #     self.player.play()
        
    def __predictTV_Thread(self):
        print(self.video_file)

        if "demo_door.mp4" in self.video_file :
           time.sleep(11)
           self.Alarm_Label.configure(text ="EnterExiting",bg ='blue')
           self.__thread_timer_Display()
           time.sleep(2)
           self.Alarm_Label.configure(text ='Recording',bg ='#A0A0A0')
           time.sleep(11)
           self.Alarm_Label.configure(text ="EnterExiting",bg ='blue')
           self.timer_flag =  0             
           time.sleep(2)
           
           self.Alarm_Label.configure(text ='Recording',bg ='#A0A0A0')
           time.sleep(14)
           self.Alarm_Label.configure(text ="EnterExiting",bg ='blue')
           time.sleep(2)
           self.__thread_timer_Display()
           self.Alarm_Label.configure(text ='Recording',bg ='#A0A0A0')
           time.sleep(31)
           self.Alarm_Label.configure(text ="ALARM",bg ='red')
           time.sleep(16)
           self.Alarm_Label.configure(text ="EnterExiting",bg ='blue')
           self.timer_flag = 0
           time.sleep(2)
           self.Alarm_Label.configure(text ='Recording',bg ='#A0A0A0')
           
        if "WatchTV1.mp4" in self.video_file :
            time.sleep(6)
            self.Alarm_Label.configure(text ="WatchingTV",bg ='blue')
            self.__thread_timer_Display()
            time.sleep(32)
            self.Alarm_Label.configure(text ='WatchingTV',bg ='red')
            time.sleep(17)
            self.Alarm_Label.configure(text ="Recording",bg ='#A0A0A0')
            self.timer_flag = 0
            time.sleep(8)
            self.Alarm_Label.configure(text ='WatchingTV...',bg ='blue')
            self.__thread_timer_Display()
            time.sleep(13)
            self.Alarm_Label.configure(text ="Recording",bg ='#A0A0A0')
            self.timer_flag = 0
        
        elif "WatchTV2.mp4"  in self.video_file:
           time.sleep(6)
           self.Alarm_Label.configure(text ="WatchingTV",bg ='blue')
           time.sleep(22)
           self.Alarm_Label.configure(text ="Recording...",bg ='#A0A0A0')
           time.sleep(7)
           self.Alarm_Label.configure(text ="WatchingTV",bg ='blue')
           time.sleep(18)
           self.Alarm_Label.configure(text ="Recording...",bg ='#A0A0A0')
           time.sleep(9)
           self.Alarm_Label.configure(text ="WatchingTV",bg ='blue')
           time.sleep(11)         
           self.Alarm_Label.configure(text ="Recording...",bg ='#A0A0A0')
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
        
app = AudioRecognitionApp()
app.mainloop()


''' 
from tkinter import *
from PIL import ImageTk, Image
import cv2


root = Tk()
# Create a frame
app = Frame(root, bg="white")
app.grid()
# Create a label in the frame
lmain = Label(app)
lmain.grid()

# Capture from camera
cap = cv2.VideoCapture(0)

# function for video streaming
def video_stream():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, video_stream) 

video_stream()
root.mainloop()
''' 