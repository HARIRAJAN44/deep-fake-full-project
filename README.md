# Deepfake Detection System

A comprehensive, multi-model ensemble system designed to detect potential deepfake manipulation in images.

## Overview
This project utilizes an ensemble approach by running five distinct deep learning models to analyze different aspects of an input image:
* **Color Mismatch Detection**
* **Face Shape Analysis**
* **Texture Analysis**
* **Eye Blink Detection**
* **Glitch Detection**

A final verdict is reached based on majority voting across all five models.

## Features
- **Multi-Model Analysis:** High reliability through ensemble classification.
- **User-Friendly GUI:** Simple desktop interface for selecting and testing images.
- **Clear Verdicts:** Provides per-model results and a final "Real" or "Fake" classification.

## Prerequisites
- Python 3.8+
- TensorFlow
- OpenCV
- Pillow
- Tkinter

## Installation
1. Clone this repository.
2. Ensure you have the required libraries installed:
   ```bash
   pip install tensorflow opencv-python pillow numpy
   ```
3. Ensure all model files (`.h5`) are located in the `main/model_train/` directory as expected by `main.py`.

## How to Run
To launch the detector GUI:
```bash
python main/main.py
```

## Project Structure
- `main/main.py`: Main entry point for the desktop GUI.
- `main/model_train/`: Directory for trained model files.
- `main/webapp/`: [Under Development] Web interface templates.


