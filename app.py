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
- Daniela D. Rabang

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

# Define the class names for fashion_MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load example images for each category
example_images = {
    'T-shirt/top': 'FashionMNIST/T-shirt_top/T-shirt_top_1.jpg',
    'Trouser': 'FashionMNIST/Trouser/Trouser_1.jpg',
    'Pullover': 'FashionMNIST/Pullover/Pullover_1.jpg',
    'Dress': 'FashionMNIST/Dress/Dress_1.jpg',
    'Coat': 'FashionMNIST/Coat/Coat_1.jpg',
    'Sandal': 'FashionMNIST/Sandal/Sandal_1.jpg',
    'Shirt': 'FashionMNIST/Shirt/Shirt_1.jpg',
    'Sneaker': 'FashionMNIST/Sneaker/Sneaker_1.jpg',
    'Bag': 'FashionMNIST/Bag/Bag_1.jpg',
    'Ankle boot': 'FashionMNIST/Ankle_boot/Ankle_boot_1.jpg'
}

model = load_model()

# Streamlit app
st.title("Fashion Item Classification")
st.write("Upload an image to classify the type of fashion item.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def import_and_predict(image_data, model):
    size = (28, 28)  # FashionMNIST images are 28x28
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image.convert('L'))  # Convert image to grayscale
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
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
    confid
