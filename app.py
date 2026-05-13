from flask import *
from flask import Flask, render_template, request, redirect, session

from flask_sqlalchemy import SQLAlchemy

import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout


# Flask app
app = Flask(__name__)

# Secret key
app.secret_key = "satellite_secret"

# MySQL Database connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/satellite_db'

# SQLAlchemy object
db = SQLAlchemy(app)


# Upload folder
UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# User table
class User(db.Model):

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(100))

    password = db.Column(db.String(100))

    email = db.Column(db.String(100))


# Build AI model
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(45, activation='softmax')

])


# Build model
model.build((None, 224, 224, 3))

# Load trained weights
model.load_weights("mobilenet_weights.weights.h5")


# Class names
class_names = [

    'airplane',
    'airport',
    'baseball_diamond',
    'basketball_court',
    'beach',
    'bridge',
    'chaparral',
    'church',
    'circular_farmland',
    'cloud',
    'commercial_area',
    'dense_residential',
    'desert',
    'forest',
    'freeway',
    'golf_course',
    'ground_track_field',
    'harbor',
    'industrial_area',
    'intersection',
    'island',
    'lake',
    'meadow',
    'medium_residential',
    'mobile_home_park',
    'mountain',
    'overpass',
    'palace',
    'parking_lot',
    'railway',
    'railway_station',
    'rectangular_farmland',
    'river',
    'roundabout',
    'runway',
    'sea_ice',
    'ship',
    'snowberg',
    'sparse_residential',
    'stadium',
    'storage_tank',
    'tennis_court',
    'terrace',
    'thermal_power_station',
    'wetland'

]


# LOGIN PAGE
@app.route("/")
def home():

    return render_template("index.html")
@app.route("/loginpage")
def loginpage():

    return render_template("login.html")

# LOGIN FUNCTION
# Login route
# LOGIN FUNCTION
@app.route("/login", methods=["POST"])
def login():

    username = request.form["username"]

    password = request.form["password"]

    user = User.query.filter_by(
        username=username,
        password=password
    ).first()

    if user:

        session["user"] = username

        return redirect("/home")

    else:

        return "Invalid Username or Password"
    
@app.route("/home")
def home_page():

    if "user" not in session:

        return redirect("/")

    return redirect("/dashboard")


# REGISTER PAGE
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        email = request.form["email"]

        new_user = User(
            username=username,
            password=password,
            email=email
        )

        db.session.add(new_user)

        db.session.commit()

        return redirect("/")

    return render_template("register.html")


# DASHBOARD PAGE
@app.route("/dashboard")
def dashboard():

    if "user" not in session:

        return redirect("/")

    return render_template("dashboard.html")


# PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    if file.filename == "":

        return "No file selected"

    # Save uploaded image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Load image
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Predict
    prediction = model.predict(img_array)

    # Get predicted index
    predicted_index = np.argmax(prediction)

    # Get class name
    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = round(
        np.max(prediction) * 100,
        2
    )

    # Show results
    return render_template(

        "results.html",

        prediction=predicted_class,

        confidence=confidence,

        image_path=filepath

    )


# LOGOUT
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect("/")


# Run app
if __name__ == "__main__":

    app.run(debug=True)