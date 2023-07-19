import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import glob
import pickle
import requests, json
from PIL import Image
from multiprocessing import Pool

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras import layers, models

from IPython.display import clear_output
from IPython import display

print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq='epoch', options=None, **kwargs):

        # Use the parent class initializer
        super().__init__(filepath, monitor, verbose,
                 save_best_only, save_weights_only,
                 mode, save_freq, options, **kwargs)

        self.epoch_counter = 0  # Add an epoch counter

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_counter += 1  # Increment the counter
        if self.epoch_counter % 50 == 0:  # Check if it's a multiple of 50
            # Save the model only on these epochs
            super().on_epoch_end(epoch, logs)

class DiscordNotificationCallback(Callback):
    def __init__(self, webhook_url, interval=50):
        super().__init__()
        self.webhook_url = webhook_url
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            if logs is not None:
                loss = logs.get('loss')
                accuracy = logs.get('accuracy')
                val_loss = logs.get('val_loss')
                val_accuracy = logs.get('val_accuracy')
                message = f"Hilbert's Normal -> Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
                payload = {"content": message}
                headers = {"Content-Type": "application/json"}
                response = requests.post(self.webhook_url, data=json.dumps(payload), headers=headers)
                
def send_discord_message(content):
    webhook_url = 'https://discord.com/api/webhooks/1120592724123467796/KLwL2pWifliFuwOzs_Az-VSGk8n1fz2lSEEEUsmEZ-UFgpRfWXHPmlZXW3AFhsmY4FWU'

    data = {
        'content': content
    }

    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    if response.status_code != 204:
        raise ValueError(f'Request to discord returned an error {response.status_code}, the response is:\n{response.text}')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Restrict TensorFlow to only use the first GPU
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

def read_DB():
    return pd.read_csv('/home/s2316002/Project2/Dataset/KDDTrain+.csv')
def preprocess(df):
    df.drop(['difficulty'],axis=1,inplace=True)
    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], dtype='int')
    df.loc[df['label'] == "normal", "label"] = 0
    df.loc[df['label'] != 0, "label"] = 1

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

def load_image(filename):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        with Image.open(filename).convert('L') as img:
            if img is not None:
                return filename, np.array(img)

def load_images_from_folder(folder):
    pool = Pool(5)
    images = pool.map(load_image, sorted(glob.glob(os.path.join(folder, '*.[jp][np]g'))))
    pool.close()
    pool.join()
    # Sort images based on filename
    images = sorted(images, key=lambda x: int(re.findall(r'\d+', x[0].split('/')[-1])[0]))
    # Remove filenames
    images = [img for _, img in images]
    return images

def plot_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('on')  # Turn off axes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Eliminate margins
def plot_9(images):
    plt.figure(figsize=(20,20))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap='gray')
        plt.xlabel(f'Image {i}')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
send_discord_message('- Hilbert Classification Model Started -')
send_discord_message('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
folder = '/home/s2316002/Project2/Image'
images = load_images_from_folder(folder)
send_discord_message('- Loading Data -')
df = read_DB()
df = preprocess(df)
imageset = np.array(images)
print(imageset.shape)

webhook_url = "https://discord.com/api/webhooks/1120592724123467796/KLwL2pWifliFuwOzs_Az-VSGk8n1fz2lSEEEUsmEZ-UFgpRfWXHPmlZXW3AFhsmY4FWU"
discord_callback = DiscordNotificationCallback(webhook_url)
checkpoint_callback = CustomModelCheckpoint('model.{epoch:02d}/500 - {val_loss:.2f}.h5')

y = df[63:].label
y
send_discord_message('- Processing Data -')
divided = int(imageset.shape[0]*(7/10))
X_train = imageset[:divided]
y_train = y[:divided].astype('int')
X_test = imageset[divided:]
y_test = y[divided:].astype('int')

X_train.shape

X_train = X_train.reshape(X_train.shape[0], 80, 96, 1)
X_test = X_test.reshape(X_test.shape[0], 80, 96, 1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

input_shape = X_train.shape[1:]
num_class = 1
input_shape
send_discord_message('- Creating model -')
model = models.Sequential([
    layers.Conv2D(16, 3, padding = 'same', activation= 'relu', input_shape = input_shape),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation= 'relu' ),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation= 'relu' ),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(num_class)
])

model.compile(optimizer = 'adam',
             loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
             metrics =['accuracy','mse'])
model.build()

model.summary()
send_discord_message(f'{model.summary()}')
send_discord_message('- Training -')
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=1000,
                    callbacks=[checkpoint_callback, discord_callback])

model.save('/home/s2316002/Project2/Model/normal_hilbert.h5')