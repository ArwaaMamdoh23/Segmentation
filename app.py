import os
import numpy as np
import tensorflow as tf
import rasterio
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_path = "deeplabv3_model.h5"  
model = load_model(model_path, compile=False)

# Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess_image(image_path, selected_channels=list(range(12)), target_size=(256, 256)):
    with rasterio.open(image_path) as src:
        image = src.read()
    
    image_selected = image[selected_channels, :, :]
    image_selected = (image_selected - np.min(image_selected)) / (np.max(image_selected) - np.min(image_selected) + 1e-8)
    
    image_resized = np.array([np.array(Image.fromarray(img).resize(target_size, Image.BILINEAR)) for img in image_selected])
    image_resized = np.moveaxis(image_resized, 0, -1)
    
    return np.expand_dims(image_resized, axis=0).astype(np.float32)  # Add batch dimension

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file is in request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        image_path = "temp.tif"
        file.save(image_path)
        
        # Preprocess image
        processed_image = preprocess_image(image_path)
        
        # Predict
        prediction = model.predict(processed_image)[0, :, :, 0]  # Remove batch dimension
        
        # Convert to binary mask
        prediction_binary = (prediction > 0.5).astype(np.uint8)

        return jsonify({"message": "Prediction successful", "prediction": prediction_binary.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
