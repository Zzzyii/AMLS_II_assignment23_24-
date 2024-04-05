
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2

from tensorflow import keras
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import clone_model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import seaborn as sns

from A.task import *
# ======================================================================================================================
# Data preprocessing
import os
for dirname, _, filenames in os.walk(../Datasets):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read four datasets related to the Bengali language.
train_data = pd.read_csv('../Datasets/bengaliai-cv19/train.csv')
test_data = pd.read_csv('../Datasets/bengaliai-cv19/test.csv')
class_map = pd.read_csv('../Datasets/bengaliai-cv19/class_map.csv')
sample_submission = pd.read_csv('../Datasets/bengaliai-cv19/sample_submission.csv')

# Show the first 5 rows of the dataset.
train_data.head()
test_data.head()
sample_submission.head()
class_map.head()

#print(f'Size of training data: {train_data.shape}')
#print(f'Size of test data: {test_data.shape}')
#print(f'Size of class map: {class_map.shape}')

# Exploratory Data Analysis
# Show most used top n data in datasets

print(f'Number of unique grapheme roots: {train_data["grapheme_root"].nunique()}')
print(f'Number of unique vowel diacritic: {train_data["vowel_diacritic"].nunique()}')
print(f'Number of unique consonant diacritic: {train_data["consonant_diacritic"].nunique()}')

# Show most used top 10 Grapheme Roots in training set.
top_10_roots = get_n(train_data, 'grapheme_root', 10)
top_10_roots

# Show vowel Diacritic in taining data
top_5_vowels = get_n(train_data, 'vowel_diacritic', 11)
top_5_vowels

#Show consonant Diacritic in training data
top_5_consonants = get_n(train_data, 'consonant_diacritic', 5)
top_5_consonants

# Calculate the number of different data values
plot_count('grapheme_root', 'grapheme_root (top 10)', train_data, size=4)
plot_count('vowel_diacritic', 'vowel_diacritic', train_data, size=3)
plot_count('consonant_diacritic', 'consonant_diacritic', train_data, size=3)

train_data = train_data.drop(['grapheme'], axis=1, inplace=False)
train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']] = train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].astype('uint8')

IMG_SIZE=64
N_CHANNELS=1

# ======================================================================================================================
# Create model
model = build_resnet34(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=1000)
#model.summary()


# Plot the structure of the model
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set a learning rate annealer
lr_re_root = ReduceLROnPlateau(monitor='root_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
lr_re_vowel = ReduceLROnPlateau(monitor='vowel_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
lr_re_consonant = ReduceLROnPlateau(monitor='consonant_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

batch_size = 256
epochs = 1

HEIGHT = 137
WIDTH = 236

#Training 
histories = []
i=0
train_set = pd.merge(pd.read_parquet(f'../Datasets/bengaliai-cv19/train_image_data_{i}.parquet'), train_data, on='image_id').drop(['image_id'], axis=1)


fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))# Show samples of current training dataset
count=0
for row in ax:
    for col in row:
        col.imshow(resize(train_set.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1).iloc[[count]], need_progress_bar=False).values.reshape(-1).reshape(IMG_SIZE, IMG_SIZE).astype(np.float64))
        count += 1
plt.show()

X_train = train_set.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
X_train = resize(X_train)/255

X_train = X_train.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

Y_train_root = pd.get_dummies(train_set['grapheme_root']).values
Y_train_vowel = pd.get_dummies(train_set['vowel_diacritic']).values
Y_train_consonant = pd.get_dummies(train_set['consonant_diacritic']).values

print(f'Training images: {X_train.shape}')
print(f'Training labels root: {Y_train_root.shape}')
print(f'Training labels vowel: {Y_train_vowel.shape}')
print(f'Training labels consonants: {Y_train_consonant.shape}')

# Divide the data into training and validation set
x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
del train_set
del X_train
del Y_train_root, Y_train_vowel, Y_train_consonant

# Data augmentation for creating more training data
datagen = DataGenerator(
    featurewise_center=False,  
    samplewise_center=False,  
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False,  
    rotation_range=8,  
    zoom_range = 0.15, 
    width_shift_range=0.15,  
    height_shift_range=0.15, 
    horizontal_flip=False,  
    vertical_flip=False)  


# This will just calculate parameters required to augment the given data. This won't perform any augmentations
datagen.fit(x_train)

# Fit the model
history = model.fit_generator(datagen.flow(x_train, {'root': y_train_root, 'vowel': y_train_vowel, 'consonant': y_train_consonant}, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test, [y_test_root, y_test_vowel, y_test_consonant]),
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[lr_re_root, lr_re_vowel, lr_re_consonant])

histories.append(history)

# Delete to reduce memory usage
del x_train
del x_test
del y_train_root
del y_test_root
del y_train_vowel
del y_test_vowel
del y_train_consonant
del y_test_consonant
gc.collect()


# ======================================================================================================================
## Print out your results

#Iterative plotting of accuracy and loss for training and validation sets.
train_loss(histories[0], epochs, 'The loss of Training Set')
train_acc(histories[0], epochs, 'The accuracy of Training Set')
val_loss(histories[0], epochs, 'The loss of Validation Set')
val_acc(histories[0], epochs, 'The accuracy of Validation Set')

del histories
gc.collect()

#The test data is predicted and the predictions are stored in a list called target.
model_result = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] 
row_id=[] 
for i in range(4):
    df_test_img = pd.read_parquet('../Datasets/bengaliai-cv19/test_image_data_{}.parquet'.format(i))
    df_test_img.set_index('image_id', inplace=True)

    X_test = resize(df_test_img, need_progress_bar=False)/255
    X_test = X_test.values.reshape(-1, IMG_SIZE, IMG_SIZE, N_CHANNELS)

    predict = model.predict(X_test)

    for i, p in enumerate(model_result):
        model_result[p] = np.argmax(predict[i], axis=1)

    for k,id in enumerate(df_test_img.index.values):
        for i,comp in enumerate(components):
            id_sample=id+'_'+comp
            row_id.append(id_sample)
            target.append(model_result[comp][k])
    del df_test_img
    del X_test
    gc.collect()

#Store the predictions in a file called submission.csv and print out the first 5 lines of data.
df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target':target
    },
    columns = ['row_id','target']
)
df_sample.to_csv('submission.csv',index=False)
df_sample.head()