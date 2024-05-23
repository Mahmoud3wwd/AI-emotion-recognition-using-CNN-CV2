import os

import seaborn as sns

sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization, \
    Input, GlobalAveragePooling2D
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import warnings

warnings.filterwarnings("ignore")

print('modules loaded')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

picture_size = 48
expression = 'happy'
folder_path = 'images3/images'

plt.figure(figsize=(5, 5))
# Get the list of images in the 'happy' folder
img_list = os.listdir(os.path.join(folder_path, "train", expression))

if len(img_list) >= 1:
    plt.subplot(3, 3, 1)
    img = load_img(os.path.join(folder_path, "train", expression, img_list[0]),
                   target_size=(picture_size, picture_size))
    plt.imshow(img)
    plt.show()
else:
    print(f"Not enough images in the folder: {expression}")

folder_path = 'images3/images'

# Make sure the 'train' and 'validation' directories exist
print(f"Folder path: {folder_path}")
print(f"Train directory exists: {os.path.exists(os.path.join(folder_path, 'train'))}")
print(f"Validation directory exists: {os.path.exists(os.path.join(folder_path, 'validation'))}")

# Define picture size
picture_size = 48

# Define batch size
batch_size = 128

# Create image data generators
datagen_train = ImageDataGenerator(rescale=1 / 255.,
                                   horizontal_flip=True, )
datagen_val = ImageDataGenerator(rescale=1 / 255., )

# Load the train and validation data
train_set = datagen_train.flow_from_directory(
    os.path.join(folder_path, 'train'),
    target_size=(picture_size, picture_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

test_set = datagen_val.flow_from_directory(
    os.path.join(folder_path, 'validation'),
    target_size=(picture_size, picture_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

no_of_classes = 5

model = Sequential()

# 1st CNN layer
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd CNN layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd CNN layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())

# Fully connected 1st layer
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Dense(no_of_classes, activation='softmax'))

checkpoint = ModelCheckpoint(
    "./model.keras",  # Use '.keras' extension instead of '.h5'
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
)


def scheduler(epoch, lr):
    if 25 < epoch < 28:
        return lr * 0.7
    else:
        return lr


scheduler_lr = keras.callbacks.LearningRateScheduler(scheduler)

callbacks_list = [checkpoint, scheduler_lr]

epochs = 50

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0009),
    metrics=['accuracy']
)
model.summary()

history = model.fit(
    x=train_set,  # Pass the generator directly to the 'x' argument
    steps_per_epoch=train_set.n // train_set.batch_size,
    epochs=epochs,
    validation_data=test_set,  # Pass the generator directly to the 'validation_data' argument
    validation_steps=test_set.n // test_set.batch_size,
    callbacks=callbacks_list
)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
