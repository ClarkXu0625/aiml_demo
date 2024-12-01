from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
import joblib
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your pre-trained model
MODEL_PATH = 'model/random_forest_iris.pkl'
#model = pickle.load(open(MODEL_PATH, 'rb'))
# Load the model
model = joblib.load(MODEL_PATH)

# Route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and inference
@app.route('/upload', methods=['POST'])
def upload():
    # Retrieve gender and age
    gender = request.form.get('gender')
    age = request.form.get('age')
    
    if not gender or not age:
        return jsonify({"error": "Gender and age are required"}), 400
    
    # Encode gender (e.g., male = 0, female = 1)
    gender_encoded = 0 if gender.lower() == 'male' else 1
    
    # Process images
    images = []
    for key in ['tongue', 'right_fingernail', 'left_fingernail', 'left_palm', 'right_palm', 'left_eye', 'right_eye']:
        file = request.files.get(key)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            image = Image.open(filepath).resize((128, 128))  # Resize for example
            image = np.array(image) / 255.0  # Normalize pixel values
            images.append(image.flatten())  # Flatten for model input
    
    if len(images) < 7:
        return jsonify({"error": "All 7 images are required"}), 400
    
    # Combine image data with gender and age
    input_data = np.array(images).flatten()
    input_features = np.append(input_data, [gender_encoded, float(age)])  # Add gender and age
    
    # convert image average rgb values
    input_features = preprocess(input_features)
    
    # Perform inference
    hemoglobin_level = model.predict(input_features)[0]
    
    return jsonify({"hemoglobin_level": hemoglobin_level})


def preprocess(input_features):
    return input_features

def prediction(model, input):
    pass

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)