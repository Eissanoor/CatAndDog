import sys
import json
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Suppress TensorFlow logging output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def predict_image(image_path, model_path):
    try:
        # Load the model
        model = keras.models.load_model(model_path)
        
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction - suppress progress bar
        predictions = model.predict(img_array, verbose=0)
        
        # Process the prediction (assuming binary classification)
        score = float(predictions[0][0])
        
        # Return result as class and confidence
        if score > 0.5:
            label = "dog"
            confidence = score
        else:
            label = "cat"
            confidence = 1 - score
            
        result = {
            "class": label,
            "confidence": float(confidence)
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Required arguments: image_path model_path"}))
        sys.exit(1)
        
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    result = predict_image(image_path, model_path)
    # Ensure only clean JSON is printed to stdout
    print(json.dumps(result)) 