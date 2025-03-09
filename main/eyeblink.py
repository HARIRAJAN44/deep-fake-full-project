import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Reshape
from tensorflow.keras.optimizers import Adam

# ✅ Define dataset path
DATASET_PATH = "archive/Dataset/"

# ✅ Ensure dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset folder not found at {DATASET_PATH}")
    exit()

print(f"✅ Dataset found at {DATASET_PATH}")

# ✅ Load dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
validation_data = valid_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ✅ Define Model (CNN + LSTM for Eye Blink Analysis)
model = Sequential([
    Flatten(input_shape=(224, 224, 3)),
    Dense(128, activation='relu'),
    Reshape((128, 1)),
    LSTM(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# ✅ Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train model
print("🚀 Training Eye Blink Model...")
model.fit(train_data, epochs=10, validation_data=validation_data)

# ✅ Save model
MODEL_PATH = "model_train/eye_blink_model.h5"
model.save(MODEL_PATH)
print(f"✅ Eye Blink Model Saved Successfully at {MODEL_PATH}")
