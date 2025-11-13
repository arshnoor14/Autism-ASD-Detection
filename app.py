from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from scripts.vectorizer_module import BERTVectorizer


app = Flask(__name__)

# Load model and scaler with error handling for the quiz-based model
try:
    model = joblib.load('models\logistic_model.pkl')
    scaler = joblib.load('models\scaler.pkl')
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Please train the model first.")
    model = None
    scaler = None

# Load the CNN model for image classification
cnn_model = tf.keras.models.load_model('autism_classification_model.keras')
IMG_HEIGHT = 180
IMG_WIDTH = 180

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/contact')
def contact():
    return render_template('contact_us.html')

@app.route('/test')
def test():
    questions = [
        "A1. Do you find it difficult to maintain eye contact with people?",
        "A2. Do you have trouble understanding social cues?",
        "A3. Do you prefer routines and dislike changes?",
        "A4. Do you get overwhelmed by loud noises or bright lights?",
        "A5. Do you find it challenging to start or maintain conversations?",
        "A6. Do you engage in repetitive behaviors?",
        "A7. Do you feel anxious in social situations?",
        "A8. Do you have intense interests in specific topics?",
        "A9. Do you struggle with understanding others' emotions?",
        "A10. Do you have difficulty making or keeping friends?"
    ]
    return render_template('index.html', questions=questions)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "Error: Prediction model not available. Please contact administrator."
    
    try:
        # Extract features from the form
        features = [
            *[float(request.form[f'q{i}']) for i in range(1, 11)],  # A1-A10 Scores
            float(request.form['age']),  # Age
            1 if request.form['gender'].lower() == "male" else 0,  # Gender
            1 if request.form.get('jaundice', '').lower() == "yes" else 0 # Jaundice
            # 1 if request.form.get('family_history', '').lower() == "yes" else 0  # Family History
        ]

        # Scale and predict
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(features_scaled)[0]

        return render_template('result.html', 
                            prediction="Autism Detected" if prediction == 1 else "No Autism Detected")

    except Exception as e:
        return f"Error processing your request: {str(e)}"

@app.route('/cnn_input')
def cnn_input():
    return render_template('cnn_input.html')  # A new page to upload image

@app.route('/cnn_predict', methods=['POST'])
def cnn_predict():
    if 'image' not in request.files:
        return "No image file provided"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Preprocess the image for CNN prediction
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        prediction = cnn_model.predict(img_array)[0]
        result = "Autism Detected" if prediction >= 0.5 else "No Autism Detected"

        return render_template('cnn_result.html', prediction=result)

text_model = joblib.load("models/autism_model.pkl")
vectorizer = BERTVectorizer.load("models/bert_vectorizer.pkl")

@app.route('/text_input')
def text_input():
    return render_template('text_input.html')

@app.route('/text_predict', methods=['POST'])
def text_predict():
    user_input = request.form['user_input']
    if not user_input.strip():
        return render_template('text_result.html', prediction="Please enter some text.")

    embedding = vectorizer.transform([user_input])
    prediction = text_model.predict(embedding)[0]
    probability = text_model.predict_proba(embedding)[0][prediction]

    label = "Autistic traits likely detected" if prediction == 1 else "Autistic traits unlikely"
    result = f"{label} (Confidence: {probability*100:.2f}%)"

    return render_template('text_result.html', prediction=result)


@app.route('/submit_contact', methods=['POST'])
def submit_contact():
    name    = request.form['name']
    email   = request.form['email']
    message = request.form['message']
    # TODO: store or email the message
    return redirect(url_for('homepage'))

if __name__ == "__main__":
    app.run(debug=True)
