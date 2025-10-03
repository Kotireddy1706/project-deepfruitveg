import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------------
# Configuration
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # Folder to save uploaded images
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = os.path.join('/Users/kotireddy/Desktop/project Deepfruitveg/notebook/fruit.h5')
model = load_model(MODEL_PATH)

# Define your class labels (match the order used in training)
CLASS_NAMES = ['Apple', 'Banana', 'Carrot', 'Tomato', 'Spinach']  # Update according to your dataset

# -------------------------------
# Helper Functions
# -------------------------------
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(file_path):
    """Load an image, preprocess it, and predict using the trained model."""
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    class_name = CLASS_NAMES[class_idx]
    confidence = float(np.max(predictions))

    return {'name': class_name, 'confidence': f"{confidence*100:.2f}%"}

# -------------------------------
# Routes for Pages
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/gallery.html')
def gallery():
    return render_template('gallery.html')

@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')

@app.route('/gallery-single.html')
def gallery_single():
    return render_template('gallery-single.html')

# -------------------------------
# Prediction API
# -------------------------------
@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Run prediction
        result = predict_image(file_path)
        return jsonify(result)

    return jsonify({'error': 'File type not allowed'}), 400

# -------------------------------
# Run the Flask Application
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
