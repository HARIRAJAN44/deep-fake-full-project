import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get absolute path of the dataset folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of dataset.py
DATASET_PATH = os.path.join(BASE_DIR, "archive/dataset/")  # Ensure correct dataset location

# Verify dataset exists before loading
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset folder not found at {DATASET_PATH}")
    exit()

print(f"✅ Dataset found at {DATASET_PATH}")

IMG_SIZE = (224, 224)  # Resize images
BATCH_SIZE = 32  # Reduce if memory issue occurs

# Define ImageDataGenerator for training (data augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

valid_test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize validation & test data

# Load training data
train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),  # ✅ This should be correct now
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load validation data
validation_data = valid_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Load test data
test_data = valid_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("✅ Dataset successfully loaded without memory issues!")
