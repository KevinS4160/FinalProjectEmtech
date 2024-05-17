# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page config
st.set_page_config(page_title="Emerging Technology 2 in CpE", layout="wide")

# Title and student details
st.title("Emerging Technology 2 in CpE")
st.markdown("""
Name:
- Kevin Roi A. Sumaya
- Daniella Rabang

Course/Section: CPE019/CPE32S5

Date Submitted: May 17, 2024
""")

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    with open("cnn_model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("cnn_model.h5")
    loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model

# Define the class names (Update this according to your waste classification labels if necessary)
class_names = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

# Load example images for each category
example_images = {
    'Cardboard': 'RealWaste/Cardboard/Cardboard_1.jpg',
    'Food Organics': 'RealWaste/Food Organics/Food Organics_1.jpg',
    'Glass': 'RealWaste/Glass/Glass_1.jpg',
    'Metal': 'RealWaste/Metal/Metal_1.jpg',
    'Miscellaneous Trash': 'RealWaste/Miscellaneous Trash/Miscellaneous Trash_1.jpg',
    'Paper': 'RealWaste/Paper/Paper_1.jpg',
    'Plastic': 'RealWaste/Plastic/Plastic_1.jpg',
    'Textile Trash': 'RealWaste/Textile Trash/Textile Trash_1.jpg',
    'Vegetation': 'RealWaste/Vegetation/Vegetation_1.jpg'
}

model = load_model()

# Streamlit app
st.title("Waste Classification")
st.write("Upload an image to classify the type of waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    return prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = import_and_predict(image, model)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

# Displaying example images for each category
st.write("## Example Images by Category")
for label, path in example_images.items():
    image = Image.open(path)
    st.image(image, caption=f'Example of {label}', use_column_width=True)
