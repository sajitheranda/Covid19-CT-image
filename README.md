# COVID-19 CT Image Classification

## Overview
This project implements a deep learning pipeline for classifying CT images into COVID-19 positive and non-COVID categories. The workflow includes data preparation, model development, performance evaluation using data augmentation, and transfer learning.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Initial Model Development](#initial-model-development)
3. [Model Performance Using Data Augmentation](#model-performance-using-data-augmentation)
4. [Transfer Learning](#transfer-learning)
5. [Results](#results)

---

## 1. Data Preparation

### 1.1 Import Dependencies
The project uses the following libraries:
```python
import os
import glob
from sklearn.model_selection import train_test_split

import PIL
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
```

### 1.2 Generate Labels and Create Datasets
The dataset includes CT images labeled as COVID positive and non-COVID. It is split into training, validation, and test sets:
```python
path = '/kaggle/input/covidct/'
pos_files = glob.glob(os.path.join(path, "CT_COVID", '*.*'))
neg_files = glob.glob(os.path.join(path, 'CT_NonCOVID', '*.*'))

images = pos_files + neg_files
labels = np.array([1] * len(pos_files) + [0] * len(neg_files))

images_train, images_test, y_train, y_test = train_test_split(
    images, labels, shuffle=True, test_size=0.2, random_state=123
)
```

### Dataset Distribution
```python
plt.title('Distribution of Labels')
plt.bar(['Positive', 'Negative'], [len(pos_files), len(neg_files)])
plt.show()
```

### Create Directories
Organize images into appropriate directories for training and testing:
```python
train_dir = os.path.join('/kaggle/working/train')
test_dir = os.path.join('/kaggle/working/test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
```

---

## 2. Initial Model Development

### 2.1 Define Model Architecture
The project uses a simple Convolutional Neural Network (CNN):
```python
model = Sequential([
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

### 2.2 Model Summary
```python
model.summary()
```

### 2.3 Train the Model
```python
history = model.fit(
    train_generator,
    steps_per_epoch=len(images_train) // 128,
    epochs=15,
    validation_data=test_generator,
    validation_steps=len(images_test) // 128
)
```

---

## 3. Model Performance Using Data Augmentation
The `ImageDataGenerator` class is used to augment the training data:
```python
train_data_gen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=128,
    class_mode='binary',
    shuffle=True
)
```

---

## 4. Transfer Learning
Fine-tuning a pre-trained model for better performance on the small dataset is the next step in the pipeline (details to be added).

---

## 5. Results

### 5.1 Training and Validation Accuracy and Loss
The model's accuracy and loss over the epochs are visualized below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/6b06d1fe-71e7-4cdb-b42d-5a91803f1e96" alt="Gas Cylinder Presentation" height="400">
  <p><em>Figure 1: Training and Validation Accuracy and Loss </em></p>
</div>


### 5.2 Confusion Matrix
The confusion matrix highlights the classification performance:

<div align="center">
  <img src="https://github.com/user-attachments/assets/6b06d1fe-71e7-4cdb-b42d-5a91803f1e96" alt="Gas Cylinder Presentation" height="400">
  <p><em>Figure 2: Training and Validation Accuracy and Loss </em></p>
</div>


### 5.3 Dataset Distribution
The distribution of positive and negative samples is illustrated below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/6b06d1fe-71e7-4cdb-b42d-5a91803f1e96" alt="Gas Cylinder Presentation" height="400">
  <p><em>Figure 3: Training and Validation Accuracy and Loss </em></p>
</div>

---

## References
- Dataset: [Kaggle COVID CT Dataset](https://www.kaggle.com)
- Frameworks: TensorFlow, Keras
