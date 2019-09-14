import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Lambda, MaxPooling2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import horovod.tensorflow.keras as hvd
import datetime
from tensorflow.keras import backend as K

hvd.init()

opt = tf.optimizers.Adam(0.001* hvd.size())
opt = hvd.DistributedOptimizer(opt)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


X_train = np.load("X.npy",allow_pickle=True)
y_train = np.load("y.npy",allow_pickle=True)
X_valid = np.load("X_valid.npy",allow_pickle=True)
y_valid = np.load("y_valid.npy",allow_pickle=True)
X_test = np.load("X_valid.npy",allow_pickle=True)
y_test = np.load("y_valid.npy",allow_pickle=True)

num_test_data = 200
nstep = 10
epochs = 10
num_train_data = len(X_train)
num_valid_data = len(X_valid) - num_test_data

loss = tf.keras.losses.BinaryCrossentropy()



X = []
for i in range(num_train_data):
   temp = tf.image.resize(X_train[i],(100,100))
   X.append(temp)
X_train = tf.convert_to_tensor(X,dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train,dtype=tf.int32)
Dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
Dataset = Dataset.shuffle(num_train_data)
Dataset = Dataset.batch(int(num_train_data/nstep))

X = []
y = []
for i in range(num_valid_data):
   temp = tf.image.resize(X_valid[i],(100,100))
   X.append(temp)
   y.append(y_valid[i])
X_valid = tf.convert_to_tensor(X,dtype=tf.float32)
y_valid = tf.convert_to_tensor(y,dtype=tf.int32)
Validset = tf.data.Dataset.from_tensor_slices((X_valid,y_valid))
Validset = Validset.shuffle(num_valid_data)
Validset = Validset.batch(int(num_valid_data/nstep))

X = []
y = []
for i in range(num_test_data):
   temp = tf.image.resize(X_test[i+num_valid_data],(100,100))
   X.append(temp)
   y.append(y_test[i+num_valid_data])
X_test = tf.convert_to_tensor(X,dtype=tf.float32)
y_test = tf.convert_to_tensor(y,dtype=tf.int32)
Testset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
Testset = Testset.shuffle(num_test_data)
Testset = Testset.batch(int(num_test_data/nstep))

reg = tf.keras.regularizers.l2(l=0.01)

class Train(Sequential):
  def __init__(self):
    super(Train,self).__init__()
    self.d1 = Conv2D(64,(5,5), activation='relu',kernel_regularizer=reg)
    self.pooling1 = MaxPooling2D(pool_size=(5,5))
    self.d3 = Conv2D(128, (3,3), activation='relu',kernel_regularizer=reg)
    self.pooling2 = MaxPooling2D(pool_size=(2,2))
    self.flatten = Flatten()
    self.drop1 = Dropout(0.5)
    self.d2 = Dense(256, activation='relu',kernel_regularizer=reg)
    self.d4 = Dense(1, activation='sigmoid')

  def call(self,x_train):
    x = self.d1(x_train)
    x = self.pooling1(x)
    x = self.d3(x)
    x = self.pooling2(x)
    x = Flatten()(x)
    x = self.drop1(x)
    x = self.d2(x)
    return self.d4(x)

nn = Train()
nn.compile(optimizer = opt,loss=loss,metrics=['accuracy'])

callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3,steps_per_epoch=nstep, verbose=1),
]
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('/scratch/awalber/tensorflow/cats_dogs2/checkpoints/checkpoint-{epoch}.h5'))
    print(hvd.size())
    print(hvd.rank())

print(callbacks)
nn.fit(x=Dataset,epochs=epochs,verbose=2,callbacks=callbacks,validation_data=Validset,use_multiprocessing=True)
print("evaluating model...")
nn.evaluate(Testset,verbose=2)
