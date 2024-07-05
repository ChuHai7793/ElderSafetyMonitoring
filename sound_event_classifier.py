from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,LSTM,TimeDistributed
from tensorflow.keras.layers import Convolution2D, MaxPooling2D,MaxPooling1D,Conv1D
from tensorflow.keras.optimizers import Adam,SGD
#from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn import metrics
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
import datetime


def reSample(data, samples):
    r = len(data)/samples #re-sampling ratio
    newdata = []
    for i in range(0,samples):
        newdata.append(data[int(i*r)])
    return np.array(newdata)
  
  
train_subjects = ['s01','s06', 's08','s07',  's09','s16', 's13', 's04', 's11', 's15', 's12', 's10']
validation_subjects = ['s02', 's03']
test_subjects = ['s05', 's17']

def get_data(path,sampleSize):
    
#    mergedActivities = ['Drinking', 'Eating', 'LyingDown', 'OpeningPillContainer', 
#                          'PickingObject', 'Reading', 'SitStill', 'Sitting', 'Sleeping', 
#                          'StandUp', 'UseLaptop', 'UsingPhone', 'WakeUp', 'Walking', 
#                          'WaterPouring', 'Writing']
    
    # specificActivities = ['Calling', 'Clapping', 'Falling', 'Sweeping', 'WashingHand', 'WatchingTV']
    specificActivities = [  'Clapping',"WashingHand",'WaterPouring']
    specificActivities = [  'Clapping']

    # specificActivities = [ 'Clapping', 'Falling','WatchingTV']

    # enteringExiting = ['Entering', 'Exiting']
    enteringExiting = []

    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_validation = []
    Y_validation = []
    
    ## Note that 'stft_257_1' contains the STFT features with specification specified in the medium article; 
    ## https://medium.com/@chathuranga.15/sound-event-classification-using-machine-learning-8768092beafc
    
    for file in os.listdir(path + 'stft_257_1/'):
        print(file)
        # if int(file.split("__")[1].split("_")[0])!=1:
        a = (np.load(path + "stft_257_1/" + file)).T
        label = file.split('_')[-1].split(".")[0]
        if(label in specificActivities):
            #if(a.shape[0]>100 and a.shape[0]<500):
            if file.split("_")[0] in train_subjects:
    #                   X_train.append(reSample(a,sampleSize))
                  X_train.append(np.mean(a,axis=0))
                  Y_train.append(label)
            elif file.split("_")[0] in validation_subjects:
                  X_validation.append(np.mean(a,axis=0))
                  Y_validation.append(label)
            else:
                  X_test.append(np.mean(a,axis=0))
                  Y_test.append(label)
                  #samples[label].append(reSample(a,sampleSize))
        elif(label in enteringExiting):
            label = "enteringExiting"
            #if(a.shape[0]>100 and a.shape[0]<500):
            if file.split("_")[0] in train_subjects:
              X_train.append(np.mean(a,axis=0))
              Y_train.append(label)
            elif file.split("_")[0] in validation_subjects:
              X_validation.append(np.mean(a,axis=0))
              Y_validation.append(label)
            else:
              X_test.append(np.mean(a,axis=0))
              Y_test.append(label)
              #samples[label].append(reSample(a,sampleSize))
        else:
            label = "other"
            #if(a.shape[0]>100 and a.shape[0]<500):
            if file.split("_")[0] in train_subjects:
                X_train.append(np.mean(a,axis=0))
                Y_train.append(label)
            elif file.split("_")[0] in validation_subjects:
                X_validation.append(np.mean(a,axis=0))
                Y_validation.append(label)
            else:
                X_test.append(np.mean(a,axis=0))
                Y_test.append(label)
                  
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_validation = np.array(X_validation)
    Y_validation = np.array(Y_validation)
    
    return X_train,Y_train,X_validation,Y_validation,X_test,Y_test
  
def print_M(conf_M):
        s = "activity,"
        for i in range(len(conf_M)):
            s += lb.inverse_transform([i])[0] + ","
        print(s[:-1])
        for i in range(len(conf_M)):
            s = ""
            for j in range(len(conf_M)):
                s += str(conf_M[i][j])
                s += ","
            print(lb.inverse_transform([i])[0],",", s[:-1])
        print()
        
        
        
def print_M_P(conf_M):
        s = "activity,"
        for i in range(len(conf_M)):
            s += lb.inverse_transform([i])[0] + ","
        print(s[:-1])
        for i in range(len(conf_M)):
            s = ""
            for j in range(len(conf_M)):
                val = conf_M[i][j]/float(sum(conf_M[i]))
                s += str(round(val,2))
                s += ","
            print(lb.inverse_transform([i])[0],",", s[:-1])
        print()        
        
# def showResult():
#   predictions = [np.argmax(y) for y in result]
#   expected = [np.argmax(y) for y in y_test]

#   conf_M = []
#   num_labels=y_test[0].shape[0]
#   for i in range(num_labels):
#       r = []
#       for j in range(num_labels):
#           r.append(0)
#       conf_M.append(r)

  

#   n_tests = len(predictions)
#   for i in range(n_tests):        
#       conf_M[expected[i]][predictions[i]] += 1

#   print_M(conf_M)
#   print_M_P(conf_M)
  
  
#Using TensorFlow backend.

cur_dir = os.getcwd()
featuresPath = cur_dir + "/STFT_features/"

a,b,c,d,e,f = get_data(featuresPath,250)
X_train,Y_train,X_validation,Y_validation,X_test,Y_test = a,b,c,d,e,f

n_samples = len(Y_train)
print("No of training samples: " + str(n_samples))
order = np.array(range(n_samples))
np.random.shuffle(order)
X_train = X_train[order]
Y_train = Y_train[order]

lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(Y_train))
y_test = to_categorical(lb.fit_transform(Y_test))
y_validation = to_categorical(lb.fit_transform(Y_validation))
num_labels = y_train.shape[1]
#No of training samples: 880
filter_size = 2


def create_model():
  model = Sequential([
    Dense(1024, input_shape=(257,)),
    Activation('relu'), 
    Dense(512),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(512),
    Activation('relu'),
    Dense(1024),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_labels),
    Activation('softmax'),
  ])
  
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

  return model


model = create_model()
model.summary()
### SAVE LAST MODEL ####

# model.save('my_model.h5')
# model.fit(X_train, y_train, batch_size=10, epochs=100,validation_data=(X_validation,y_validation))


### SAVE BEST MODEL ####
# checkpoint = ModelCheckpoint('models\\new_model.h5', verbose=1, 
#                              monitor='val_loss',save_best_only=True, mode='auto') 
 
# model.fit(X_train, y_train, batch_size=10, epochs=100,validation_data=(X_validation,y_validation), callbacks=[checkpoint], verbose=True)






