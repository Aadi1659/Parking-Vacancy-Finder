import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# allowing more memory for tensorflow, tensorflow auto use GPU if detected
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#dirs
dataset_dir_name = ['parking_dataset']
dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dataset_dir_name[0])
# print(dataset_dir)

#traning params
classes_name = ['0','1']
batch_size_train = 32
img_size = (200,200)
epochs = 10
AUTOTUNE = tf.data.AUTOTUNE   # automatically set num of parallel worker

def prepare(img):
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size[0],img_size[1],3))
    img = rescale(img)
    return img

#train and val split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.3,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size_train
)

train_ds = train_ds.map(lambda x, y: (prepare(x), y), num_parallel_calls=AUTOTUNE)
#x and y represent the features and labels of the dataset

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          dataset_dir,
          validation_split=0.3,
          subset="validation",
          seed=123,
          image_size=img_size,
          batch_size=batch_size_train)

val_ds = val_ds.map(lambda x, y: (prepare(x), y), num_parallel_calls=AUTOTUNE)

#performance configuration
#.cache() keeps the dataset for perfomance
#.prefect() simultaneous data preprocessing and training process
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)

num_classes = 2 #binary data

#making the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(img_size[0], img_size[1], 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 7, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

#compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

#summary
model.summary()

#train the model
history_model = model.fit(train_ds,
                          validation_data = val_ds,
                          epochs=epochs)

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
  plt.show()

plot_loss_curves(history=history_model)

model.save_weights('my_checkpoint')
model.save("my_model")
