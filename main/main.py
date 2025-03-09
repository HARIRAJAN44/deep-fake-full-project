import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# ✅ Load trained models
MODEL_PATHS = {
    "Color Mismatch": "model_train/color_mismatch_model.h5",
    "Face Shape": "model_train/face_shape_model.h5",
    "Texture Analysis": "model_train/texture_analysis_model.h5",
    "Eye Blink": "model_train/eye_blink_model.h5",
    "Glitch Detection": "model_train/artifact_model.h5"
}

models = {name: load_model(path) for name, path in MODEL_PATHS.items()}
print("✅ All models loaded successfully!")

# ✅ Function to preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Expand dimensions for model
    return img

# ✅ Function to predict using all models
def predict_image(img_path):
    img = preprocess_image(img_path)
    predictions = {}

    for model_name, model in models.items():
        pred = model.predict(img)[0][0]
        predictions[model_name] = "Real" if pred > 0.5 else "Fake"  # ✅ Corrected swap

    # ✅ Majority Voting Fix
    real_votes = sum(1 for result in predictions.values() if result == "Real")
    fake_votes = len(predictions) - real_votes
    final_result = "Fake" if fake_votes > real_votes else "Real"  # ✅ Corrected final result

    return predictions, final_result



# ✅ GUI Implementation
class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detector")
        self.root.geometry("500x500")
        self.root.configure(bg="black")

        # Label for Title
        self.title_label = Label(root, text="Deepfake Detector", font=("Arial", 18, "bold"), fg="white", bg="black")
        self.title_label.pack(pady=10)

        # Label for Image Display
        self.image_label = Label(root, text="No Image Selected", font=("Arial", 12), fg="white", bg="black")
        self.image_label.pack(pady=10)

        # Button to Select Image
        self.select_button = Button(root, text="Select Image", font=("Arial", 12), command=self.select_image, bg="blue", fg="white")
        self.select_button.pack(pady=10)

        # Button to Predict
        self.predict_button = Button(root, text="Predict", font=("Arial", 12), command=self.predict, bg="green", fg="white")
        self.predict_button.pack(pady=10)

        # Label to Show Result
        self.result_label = Label(root, text="", font=("Arial", 14, "bold"), fg="white", bg="black")
        self.result_label.pack(pady=10)

        # Store Selected Image Path
        self.image_path = None

    # ✅ Function to Select Image
    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = img.resize((200, 200))  # Resize image for display
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
            self.image_label.config(text="")  # Remove text when image is displayed

    # ✅ Function to Predict Image
    def predict(self):
        if not self.image_path:
            self.result_label.config(text="❌ No image selected. Please select an image!", fg="red")
            return
        
        predictions, final_result = predict_image(self.image_path)


        # Display Prediction Results
        result_text = "\n".join([f"{model}: {result}" for model, result in predictions.items()])
        result_text += f"\n\nFinal Verdict: {final_result}"

        # Update GUI with result
        self.result_label.config(text=result_text, fg="red" if final_result == "Fake" else "green")

# ✅ Run the Tkinter App
if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    root.mainloop()
