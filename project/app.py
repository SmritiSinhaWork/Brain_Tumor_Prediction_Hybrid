import cv2
import torch
from flask import Flask, render_template, request, jsonify
import os
from dmd import dmd_process
from model import HybridModel
from utils import predict

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = HybridModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route (AJAX compatible)
@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        # Get file
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        # Save file
        filename = os.path.basename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print("Saved at:", filepath)

        # Read + preprocess image
        img = cv2.imread(filepath, 0)

        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        img = cv2.resize(img, (64, 64))
        img = dmd_process(img)
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Predict
        result = predict(model, img)

        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        label = classes[result]

        # Return JSON (IMPORTANT)
        return jsonify({
            "prediction": label
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            "error": "Something went wrong"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)