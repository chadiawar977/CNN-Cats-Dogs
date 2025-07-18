from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Define class names for binary classification (0 = dog, 1 = cat)
CLASS_NAMES = ['Dog', 'Cat']

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained .keras model
try:
    model = load_model('final_model.h5' , compile = False )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(file_path):
    """Preprocess the image to be suitable for the model"""
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Flatten if model expects 8192 input shape
    if model.input_shape[-1] == 8192:
        img_array = img_array.reshape((1, -1))
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process the image
            processed_image = preprocess_image(file_path)

            # Make prediction
            prediction = model.predict(processed_image)[0][0]  # Get the single output value

            # For binary classification where 0 = dog, 1 = cat
            # The closer to 1, the more confident it's a cat
            # The closer to 0, the more confident it's a dog
            dog_probability = float(prediction)
            cat_probability = 1.0 - dog_probability

            # Determine predicted class based on threshold (0.5)
            if dog_probability >= 0.5:
                predicted_class = 'Dog'
                confidence = dog_probability
            else:
                predicted_class = 'Cat'
                confidence = cat_probability

            # Create a dictionary of class probabilities
            all_probabilities = {
                'Cat': cat_probability,
                'Dog': dog_probability
            }

            return jsonify({
                'result': predicted_class,
                'confidence': f"{confidence * 100:.2f}%",
                'all_probabilities': all_probabilities,
                'image_path': f"/static/uploads/{filename}"
            })
        except Exception as e:
            return jsonify({'error': f"Error processing image: {str(e)}"})

    return jsonify({'error': 'Invalid file type. Please upload an image (png, jpg, jpeg)'})


if __name__ == '__main__':
    app.run(debug=True)