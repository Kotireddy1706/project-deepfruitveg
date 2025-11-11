import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image

# -------------------------------------
# Flask App Configuration
# -------------------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')

# Folder to store uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------------
# Load Model and Labels
# -------------------------------------
MODEL_PATH = os.path.join('models', 'fruit_model.h5')   # Path to your model
LABELS_PATH = os.path.join('models', 'labels.txt')       # Text file containing class names

# Load model once when app starts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully from:", MODEL_PATH)
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# Load labels
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines()]
    print("✅ Labels loaded successfully:", LABELS)
else:
    print("⚠️ labels.txt not found! Please ensure it exists in /models directory.")
    LABELS = []

# -------------------------------------
# Page Routes
# -------------------------------------
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/gallery.html')
def gallery():
    return render_template('gallery.html')

@app.route('/gallery-single.html')
def gallery_single():
    return render_template('gallery-single.html')

@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

# -------------------------------------
# API Route: Handle Image Upload + Prediction
# -------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded properly!'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file to static/uploads
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Preprocess image (ensure it matches your model input)
        img = image.load_img(file_path, target_size=(64, 64))  # adjust based on your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalization

        # Make prediction
        predictions = model.predict(img_array)
        confidence = round(float(np.max(predictions)) * 100, 2)
        result_index = np.argmax(predictions)
        predicted_label = LABELS[result_index] if LABELS else "Unknown"

        print(f"🔍 Prediction: {predicted_label} ({confidence}%)")

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence,
            'filename': file.filename
        })

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({'error': str(e)}), 500

# -------------------------------------
# Run Flask App
# -------------------------------------
if __name__ == "__main__":
    # For local development
    app.run(debug=True, host="0.0.0.0", port=5000)
