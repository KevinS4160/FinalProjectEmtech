 import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np

@st.cache_resource
# Streamlit app
st.title("MNIST Digit Classifier")

# Load the model architecture from JSON
with open("/content/model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Load the model weights
model.load_weights("/content/drive/MyDrive/Colab Notebooks/model.h5")

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict the class
    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")
