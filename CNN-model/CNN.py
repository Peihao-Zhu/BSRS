import numpy as np
#import pandas as pd
import scipy.io as sio
import h5py
import scipy.io

from keras.models import *
from keras.layers import Input, merge, Conv1D,Conv2D, Dense, Flatten, Dropout, MaxPooling1D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

datas = h5py.File('trainandtest_70.mat')
# 加载 matFile 内的数据
# 假设 mat 内保存的变量为 matlabdata

trainsample = datas['trainsample']
testsample = datas['testsample']
trainlabel = datas['trainlabel']
gttestlabel = datas['testlabel']

trainlabel = np.array(trainlabel)
print(trainlabel[:16])
# Keras model with one Convolution1D layer
# unfortunately more number of covnolutional layers, filters and filters lenght 
# don't give better accuracy

print(trainsample.shape)
inputs = Input((None,70))
print(inputs.shape)

conv1 = Conv1D(128, 1,  padding='valid',bias='none' ,kernel_initializer='he_normal', name = 'Conv1D_1')(inputs)
conv1 = Conv1D(64, 1,  padding='valid', kernel_initializer='he_normal',activation = 'relu')(conv1)
dense9 = Dense(32, activation = 'relu')(conv1)
dense9 = Dropout(0.05)(dense9)
dense10 = Dense(16, activation = 'softmax')(dense9)
model = Model(inputs=inputs,output = dense10)
model.compile(optimizer = Adam(lr = 0.5*1e-4), loss = 'categorical_crossentropy',metrics=['accuracy'])
model.summary()
model_checkpoint = ModelCheckpoint('indian.hdf5', monitor='loss',verbose=1, save_best_only=True)
print('Fitting model...')
model.fit(trainsample, trainlabel, batch_size=3000, nb_epoch=300000, verbose=1, shuffle=True,validation_split=0.2, callbacks=[model_checkpoint])

cnnlayermodel = Model(input = model.input, outputs=model.get_layer('Conv1D_1').output)
conv1_output = cnnlayermodel.predict(trainsample, verbose=1)

#获得某一层的权重和偏置
weight_Conv1_1,bias_Conv1_1 = model.get_layer('Conv1D_1').get_weights()

print('trainsample', trainsample.shape)
print('conv1_output', conv1_output.shape)
print('weight_Conv1', weight_Conv1_1.shape)
print('bias_Conv1', bias_Conv1_1.shape)

scores_10=model.evaluate(testsample,gttestlabel,verbose=0)
print('Test Loss:',scores_10[0])
print('Test Accuracy:',scores_10[1])

scipy.io.savemat('TestResult_70.mat',{'testloss':scores_10[0],'testaccuracy':scores_10[1]})