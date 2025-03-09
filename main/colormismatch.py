import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# ✅ Define dataset path (Adjust this based on your dataset location)
DATASET_PATH = "archive/Dataset/"

# ✅ Ensure dataset exists before loading
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset folder not found at {DATASET_PATH}")
    exit()

print(f"✅ Dataset found at {DATASET_PATH}")

# ✅ Image settings
IMG_SIZE = (224, 224)  # Resize images
BATCH_SIZE = 32  # Adjust based on memory
EPOCHS = 10  # Number of training cycles

# ✅ Define ImageDataGenerator for data loading
train_datagen = ImageDataGenerator(rescale=1./255)
valid_test_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Load training data
train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ✅ Load validation data
validation_data = valid_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ✅ Define a Simple Neural Network Model for Color Mismatch Detection
model = Sequential([
    Flatten(input_shape=(224, 224, 3)),  # Convert 2D image to 1D
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: 0 (Real) or 1 (Fake)
])

# ✅ Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train model
print("🚀 Training Color Mismatch Model...")
model.fit(train_data, epochs=EPOCHS, validation_data=validation_data)

# ✅ Save model in `model_train/`
MODEL_PATH = "model_train/color_mismatch_model.h5"
model.save(MODEL_PATH)
print(f"✅ Color Mismatch Model Saved Successfully at {MODEL_PATH}")
