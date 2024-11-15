from flask import Flask, render_template,  redirect, url_for, flash, session
from flask import Flask, request
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from PIL import Image
import json


app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secretKey'
app.config['MONGO_URI'] = 'mongodb+srv://admin:aQ7lG5a5A4N3bSFb@cluster0.mlod4ao.mongodb.net/plantationProject'
mongo = PyMongo(app)
db = mongo.db
crop_recommendation_model_path = 'models/recommendation_Model.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# deficiency_prediction_model_path = 'models/Deficiency_prediction.pkl'
# deficiency_prediction_model = pickle.load(open(deficiency_prediction_model_path, 'rb'))
model = load_model('models/plant_disease_prediction_model.h5')

# # with open('models/class_indices.json', 'r') as f:
# #     class_indices = json.load(f)

# class_names = {v: k for k, v in class_indices.items()}

class_names={0: 'Apple___Apple_scab',
 1: 'Apple___Black_rot',
 2: 'Apple___Cedar_apple_rust',
 3: 'Apple___healthy',
 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew',
 6: 'Cherry_(including_sour)___healthy',
 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight',
 10: 'Corn_(maize)___healthy',
 11: 'Grape___Black_rot',
 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape___healthy',
 15: 'Orange___Haunglongbing_(Citrus_greening)',
 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy',
 18: 'Pepper,_bell___Bacterial_spot',
 19: 'Pepper,_bell___healthy',
 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight',
 22: 'Potato___healthy',
 23: 'Raspberry___healthy',
 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew',
 26: 'Strawberry___Leaf_scorch',
 27: 'Strawberry___healthy',
 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight',
 30: 'Tomato___Late_blight',
 31: 'Tomato___Leaf_Mold',
 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite',
 34: 'Tomato___Target_Spot',
 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus',
 37: 'Tomato___healthy'}





@app.route('/')
def index():
    return render_template("index.html")

@app.route('/nd')
def first():
    return render_template("nutrition_deficiency.html")

@app.route('/crop-recc')
def second():
    return render_template("crop_recommendation.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = mongo.db.users.find_one({'email': email})
        
        if user and check_password_hash(user['password'], password):
            session['user'] = email
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check your email and/or password.', 'danger')
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = {
            'name': name,
            'email': email,
            'password': hashed_password
        }
        app.logger.debug(f"Inserting new user: {new_user}")
        try:
            db.users.insert_one(new_user)
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            app.logger.error(f'Registration failed: {str(e)}')
            flash(f'Registration failed: {str(e)}', 'danger')
            return redirect(url_for('register'))
    return render_template("register.html")

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template("how_it_works.html")


@app.route('/services')
def services():
    return render_template('services.html')


@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

@app.route('/request')
def req():
    return render_template('request.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    # Handle the form submission logic here (e.g., save to database, send email, etc.)
    return "Message sent successfully!"


crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}


reverse_crop_dict = {v: k for k, v in crop_dict.items()}

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        T = float(request.form['temperature'])
        H = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        data = np.array([[N, P, K, T, H, ph, rainfall]])
       
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        crop_index = np.argmax(final_prediction)
        crop_name = reverse_crop_dict.get(crop_index + 1, "Unknown Crop")
        return render_template('recommend_result.html', prediction=crop_name)

# @app.route('/crop-deficiency', methods=['POST'])
# def crop_deficiency():
#     if request.method == 'POST':
#         N = float(request.form['nitrogen'])
#         P = float(request.form['phosphorous'])
#         K = float(request.form['pottasium'])
#         ph = float(request.form['ph'])
#         CT = request.form['cropname']
        
#         data = np.array([[N, P, K, ph, CT]], dtype=object)
#         my_prediction = deficiency_prediction_model.predict(data)
#         final_prediction = my_prediction[0]
# return render_template('deficiency_result.html', prediction=final_prediction)
@app.route('/disease')
def disease():
    return render_template('disease.html')


def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

  
 
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file:
        # Preprocess the uploaded image
        img_array = load_and_preprocess_image(file)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        
        # Pass prediction to disease_result.html
        return render_template('disease_result.html', prediction=predicted_class_name)
    else:
        return redirect(url_for('disease'))
    
if __name__ == "__main__":
    app.run(port=5555, debug=True)
