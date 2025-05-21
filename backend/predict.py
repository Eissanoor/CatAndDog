import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'cat_dog_other_model.h5')  # Updated model name
IMG_WIDTH = 224
IMG_HEIGHT = 224
CONFIDENCE_THRESHOLD = 0.60  # Threshold for confident predictions

# Class mapping
CLASS_MAPPING = {0: 'Cat', 1: 'Dog', 2: 'Other'}  # Added class mapping for multi-class model

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for prediction"""
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array, img

def predict_image(model, image_path):
    """Predict whether the image contains a cat, dog, or other"""
    img_array, img = load_and_preprocess_image(image_path)
    
    # Make prediction - now expecting an array with 3 probabilities
    predictions = model.predict(img_array)[0]
    
    # Get the class with highest probability
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]
    class_name = CLASS_MAPPING[class_index]
    
    # If confidence is too low, we might want to indicate that
    if confidence < CONFIDENCE_THRESHOLD:
        class_name += " (low confidence)"
    
    return class_name, confidence, img

def display_prediction(image_path, class_name, confidence, img):
    """Display the image with prediction results"""
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {class_name}\nConfidence: {confidence:.2%}")
    plt.axis('off')
    plt.show()

def select_image():
    """Open a file dialog to select an image"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    
    return file_path

def main():
    # Use file dialog to select image
    print("Please select an image file...")
    image_path = select_image()
    
    # Check if user canceled the file selection
    if not image_path:
        print("No image selected. Exiting.")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return
    
    # Load the model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make prediction
    try:
        class_name, confidence, img = predict_image(model, image_path)
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence:.2%}")
        
        # Display the image with prediction
        display_prediction(image_path, class_name, confidence, img)
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
