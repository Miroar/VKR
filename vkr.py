import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import librosa as lb
import librosa.display as lbd
import os
train=pd.read_csv('../input/ processed_audio_files/train.csv')
val=pd.read_csv('../input/processed_audio_files /val.csv')
def getFeatures(path):
    soundArr,sample_rate=lb.load(path)
    mfcc=lb.feature.mfcc(y=soundArr,sr=sample_rate)
    cstft=lb.feature.chroma_stft(y=soundArr,sr=sample_rate)
    mSpec=lb.feature.melspectrogram(y=soundArr,sr=sample_rate)
    return mfcc,cstft,mSpec
root='../input/ processed_audio_files/'
mfcc,cstft,mSpec=[],[],[]
for idx,row in val.iterrows():
    path=root + row['filename']
    a,b,c=getFeatures(path)
    mfcc.append(a)
    cstft.append(b)
    mSpec.append(c)
mfcc_val=np.array(mfcc)
cstft_val=np.array(cstft)
mSpec_val=np.array(mSpec)
root='../input/ processed_audio_files'
mfcc,cstft,mSpec=[],[],[]
for idx,row in train.iterrows():
    path=root + row['filename']
    a,b,c=getFeatures(path)
    mfcc.append(a)
    cstft.append(b)
    mSpec.append(c)
mfcc_train=np.array(mfcc)
cstft_train=np.array(cstft)
mSpec_train=np.array(mSpec)
mfcc_input=keras.layers.Input(shape=(12,259,1),name="mfcc")
x=keras.layers.Conv2D(16,2,padding='same')(mfcc_input)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(32,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(64,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(128,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
mfcc_output=keras.layers.GlobalMaxPooling2D()(x)
mfcc_model=keras.Model(mfcc_input, mfcc_output, name="mfcc")
croma_input=keras.layers.Input(shape=(12,259,1),name="chroma")
x=keras.layers.Conv2D(16,2,padding='same')(croma_input)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(32,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(64,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(128,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
croma_output=keras.layers.GlobalMaxPooling2D()(x)
croma_model=keras.Model(croma_input, croma_output, name="chroma")
mSpec_input=keras.layers.Input(shape=(128,259,1),name="melSpec")
x=keras.layers.Conv2D(16,2,padding='same')(mSpec_input)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(32,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(64,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
x=keras.layers.MaxPooling2D(pool_size=2,padding='valid')(x)
x=keras.layers.Conv2D(128,2,padding='same')(x)
x=keras.layers.BatchNormalization()(x)
x=keras.layers.Activation(keras.activations.relu)(x)
mSpec_output=keras.layers.GlobalMaxPooling2D()(x)
mSpec_model=keras.Model(mSpec_input, mSpec_output, name="melSpec")
input_mfcc=keras.layers.Input(shape=(20,259,1),name="mfccInput")
mfcc=mfcc_model(input_mfcc)
input_croma=keras.layers.Input(shape=(12,259,1),name="chromaInput")
croma=croma_model(input_croma)
input_mSpec=keras.layers.Input(shape=(128,259,1),name="melSpecInput")
mSpec=mSpec_model(input_mSpec)
concat=keras.layers.concatenate([mfcc,croma,mSpec])
hidden=keras.layers.Dropout(0.2)(concat)
hidden=keras.layers.Dense(50,activation='relu')(concat)
hidden=keras.layers.Dropout(0.2)(hidden)
hidden=keras.layers.Dense(25,activation='relu')(hidden)
hidden=keras.layers.Dropout(0.2)(hidden)
output=keras.layers.Dense(6,activation='softmax')(hidden)
net=keras.Model([input_mfcc,input_croma,input_mSpec], output, name="Net")
from keras import backend as K
net.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
K.set_value(net.optimizer.learning_rate, 0.001)

root='../input/checking/'
names=pd.read_csv('../input/checking/names.csv')
mfcc,cstft,mSpec=[],[],[]
input_array = []
for row in names.iterrows():
    path=root + row['filename']
    a,b,c =getFeatures(path)
    mfcc.append(a)
    cstft.append(b)
    mSpec.append(c)
    input_array.append(mfcc)
    input_array.append(cstft)
    input_array.append(mSpec)
    result = net.predict(input_array)
    print(result.index(max(result)))
