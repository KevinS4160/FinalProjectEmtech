# Save this as `app.py`
import streamlit as st
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image

# Streamlit app
st.title("Fashion MNIST Classifier")

# Load the model architecture from JSON
with open("cnn_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("cnn_model.h5")

# Compile the model
loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
    prediction = loaded_model.predict(img_array)
    st.write(f"Predicted Class: {np.argmax(prediction)}")
