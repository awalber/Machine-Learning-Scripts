import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os

start_time = time.time()

dir = os.environ['sc_dir']
path = dir+"/NLP/"
save_file = path+"weights"
files = ["Fake.csv","True.csv"]
all_data=True

fake_df = pd.read_csv(path+files[0],dtype=str)
true_df = pd.read_csv(path+files[1],dtype=str)
AUTOTUNE=tf.data.experimental.AUTOTUNE

if not all_data:
  num_data = 10000
  data = pd.concat([fake_df["text"][0:int(num_data/2)],true_df["text"][0:int(num_data/2)]])
  true_target = np.ones(int(num_data/2),)
  false_target = np.zeros(int(num_data/2),)

else:
  true_data = len(true_df["text"])
  fake_data = len(fake_df["text"])
  print(true_data)
  data = pd.concat([fake_df["text"][0:],true_df["text"][0:]])
  true_target = np.ones(int(true_data),)
  false_target = np.zeros(int(fake_data),)


true_target = pd.DataFrame(true_target,dtype=int,columns=["target"])
false_target = pd.DataFrame(false_target,dtype=int,columns=["target"])

target = pd.concat([false_target,true_target])
target = target.pop("target")

TAKE_SIZE = 4000
BUFFER_SIZE = int(len(target) - TAKE_SIZE)
BATCH_SIZE = 2048
NSTEP = int(BUFFER_SIZE/BATCH_SIZE)

all_labeled_data = tf.data.Dataset.from_tensor_slices((data.values,target.values))
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()

for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode,
                                       inp=[text, label],
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually:
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label


def prepare_for_training(ds, cache=True, train=True, shuffle_buffer_size=BUFFER_SIZE):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  if train:
    ds = ds.skip(TAKE_SIZE).shuffle(shuffle_buffer_size)
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)

  else:
    ds = ds.take(TAKE_SIZE)
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))
    ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


all_encoded_data = all_labeled_data.map(encode_map_fn,num_parallel_calls=AUTOTUNE)


train_data = prepare_for_training(all_encoded_data)
test_data = prepare_for_training(all_encoded_data,train=False)

#train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
#train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))
#test_data = all_encoded_data.take(TAKE_SIZE)
#test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

vocab_size = len(vocabulary_set)
vocab_size += 1

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 64))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu'))

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(2))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_data,steps_per_epoch=NSTEP,epochs=10,validation_data=test_data,verbose=2)

print(time.time() - start_time)

model.save_weights(filepath=save_file)
