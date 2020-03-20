import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from tensorflow.keras import Sequential
import glob
import pathlib
import time
from mpi4py import MPI

save_name = 'Four_node'

comm = MPI.COMM_WORLD
size = comm.Get_size() # Total number of tasks. 1 node per task => number of nodes
rank = comm.Get_rank() # List of processes [0:n-1]

time1 = time.time()
parallel_calls = 10

directory=os.environ['sc_dir']
epochs = 20
img_height =200
img_width = 200
channels = 3
logdir = directory+"/tensorflow/cats_dogs2/mpi/callbacks/"
train_dir = directory+'/tensorflow/cats_dogs2/mpi/data/train'
test_dir = directory+'/tensorflow/cats_dogs2/mpi/data/test1'
valid_dir = directory+'/tensorflow/cats_dogs2/mpi/data/valid'
data_dir = pathlib.Path(train_dir)
valid_dir = pathlib.Path(valid_dir)
test_dir = pathlib.Path(test_dir)
num_train_images = len(list(data_dir.glob('*/*.jpg')))
batch_size = int(16)
nstep = int(num_train_images/size/batch_size)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
cat_ds = tf.data.Dataset.list_files(str(data_dir/'cat/*'))
dog_ds = tf.data.Dataset.list_files(str(data_dir/'dog/*'))
list_valid = tf.data.Dataset.list_files(str(valid_dir/'*/*'))
list_test = tf.data.Dataset.list_files(str(test_dir/'*/*'))

for i in range(size):
  if rank == i:
    ds1 = cat_ds.take(int(num_train_images/size/2))
    ds2 = dog_ds.take(int(num_train_images/size/2))
    list_ds = ds1.concatenate(ds2)
reg_init = 0.0001
reg = tf.keras.regularizers.l1(reg_init)
loss = tf.keras.losses.BinaryCrossentropy()
initial_lr = 0.00001
opt = tf.keras.optimizers.Adam(lr=initial_lr)

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, '/')
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img,training=False):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=channels)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize_with_pad(img, img_height, img_width)

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=int(num_train_images)):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat infinitely and batch the dataset
  ds = ds.repeat()
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=parallel_calls)


  return ds

class Train(Sequential):
  def __init__(self):
    super(Train,self).__init__()
    self.c1 = Conv2D(32, (3,3), activation='relu',kernel_regularizer=reg,bias_regularizer=reg,input_shape=(img_height,img_width,channels))
    self.pooling1 = MaxPooling2D(pool_size=(2,2))
    self.c2 = Conv2D(64, (3,3), activation='relu',kernel_regularizer=reg,bias_regularizer=reg)
    self.pooling2 = MaxPooling2D(pool_size=(2,2))
    self.c3 = Conv2D(128, (3,3), activation='relu',kernel_regularizer=reg,bias_regularizer=reg)
    self.pooling3 = MaxPooling2D(pool_size=(2,2))
    self.c4 = Conv2D(256, (3,3), activation='relu',kernel_regularizer=reg,bias_regularizer=reg)
    self.pooling4 = MaxPooling2D(pool_size=(2,2))
    self.flatten = Flatten()
    self.d1 = Dense(256, activation='relu')
    self.drop1 = Dropout(0.5)
    self.d2 = Dense(256, activation='relu')
    self.drop2 = Dropout(0.5)
    self.d3 = Dense(2, activation='sigmoid')

  def call(self,x_train):
    x = self.c1(x_train)
    x = self.pooling1(x)
    x = self.c2(x)
    x = self.pooling2(x)
    x = self.c3(x)
    x = self.pooling3(x)
    x = self.c4(x)
    x = self.pooling4(x)
    x = Flatten()(x)
    x = self.d1(x)
    x = self.drop1(x)
    x = self.d2(x)
    x = self.drop2(x)
    return self.d3(x)

class mpicallbacks(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_time = 0
    self.start_time = 0
    self.epoch_time = 0
    
  def on_train_begin(self,epoch,logs=None):
    weights = self.model.get_weights()
    starting_weights = comm.bcast(weights)
    self.model.set_weights(starting_weights)
    
  def on_train_batch_begin(self,batch,logs=None):
    self.start_time= time.time()
    
  def on_train_batch_end(self,batch,logs=None):
    self.batch_time = time.time()-self.start_time
    self.epoch_time += self.batch_time
    
  def on_epoch_end(self,epoch,logs=None):
    total_time = self.epoch_time
    weight_time = time.time()
    weights = self.model.get_weights()
    data_list = np.asarray(weights)
    data = comm.reduce(data_list)
    if rank == 0:
      data = data/size
    new_weights = comm.bcast(data)
    self.model.set_weights(new_weights)
    data_transfer = time.time()-weight_time
    if rank == 0: 
      print("sharing weights took:",data_transfer,"seconds")
    total_time = total_time + data_transfer
    if rank == 0:
      print("Processing",(size*batch_size*nstep/total_time),"images per second across",size,"node(s)")
      print("Total time per epoch:",total_time)
    if epoch % 5 == 0 and rank == 0 and epoch !=0:
      print("Saving model weights now!")
      self.model.save_weights(logdir+save_name)
    self.epoch_time = 0


labeled_ds = list_ds.map(process_path,num_parallel_calls=parallel_calls)
valid_ds = list_valid.map(process_path,num_parallel_calls=parallel_calls)
test_ds = list_test.map(process_path,num_parallel_calls=parallel_calls)
train_ds = prepare_for_training(labeled_ds)
valid_ds = prepare_for_training(valid_ds)
test_ds = test_ds.batch(len(list(test_dir.glob('*/*.jpg'))))


model = Train()
#model.load_weights(logdir+save_name)
model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
if rank == 0:
  print("Images loaded and model compiled in",time.time()-time1,"seconds")

verbose = 2 if rank == 0 else 0
model.fit(x=train_ds,epochs=epochs,steps_per_epoch=nstep, validation_data=valid_ds, validation_steps=nstep, verbose=verbose, callbacks=[mpicallbacks()])
if rank == 0:
  print("saving final weights...")
  model.save_weights(logdir+save_name)
  loss, acc = model.evaluate(test_ds,verbose=0)
  print("Testing loss:",loss)
  print("Testing accuracy:",acc)
