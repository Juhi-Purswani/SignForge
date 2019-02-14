
# coding: utf-8

# In[1]:


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm 

TRAIN_DIR = 'data/training'
TEST_DIR = 'data/testing'
IMG_SIZE = 28
LR = 1e-3


# In[2]:


def label_img(img):
    image_label = img.split('.')
    name = image_label[0]
    # checking if the signature is forged or genuine
    #                            
    
    if name[4:7] == name[-3:]:
        return 1
  
    elif name[4:7] != name[-3:]:
        return 0


# In[ ]:


def person_img(img):
    label = img.split('.')
    k = label[0]
    return [k[-3:]]


# In[ ]:



def create_features(DIR):
    features = []
    labels = []
    for img in tqdm(os.listdir(DIR)):
        label = label_img(img)
        person = person_img(img)
       
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        features.append([img])
        labels.append(np.array(label))
    #shuffle(training_data)
    #np.save('train_data.npy', training_data)
    return features


# In[ ]:



train_features  = np.array(create_features(TRAIN_DIR))
test_features  = np.array(create_features(TEST_DIR))
train_features_reshaped = train_features.reshape(-1, 28,28, 1)
test_features_reshaped = test_features.reshape(-1, 28,28, 1)

train_features_reshaped = train_features_reshaped.astype('float32')
test_features_reshaped = test_features_reshaped.astype('float32')
train_features_reshaped = train_features_reshaped / 255.
test_features_reshaped = test_features_reshaped / 255.


# In[ ]:


def create_labels(DIR):
    labels = []
    for img in tqdm(os.listdir(DIR)):
        label = label_img(img)
        labels.append(np.array(label))
    return labels


# In[ ]:



train_labels = np.array(create_labels(TRAIN_DIR))
test_labels = np.array(create_labels(TEST_DIR))


# In[ ]:


import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# In[ ]:


train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)


# In[ ]:


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 100
num_classes = 2

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()


# In[ ]:


fit = model.fit(train_features_reshaped, train_labels_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,
                                  validation_data=(test_features_reshaped, test_labels_one_hot))

