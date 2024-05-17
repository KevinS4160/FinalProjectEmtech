import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Streamlit app
st.title("fashion_mnist")

# Define the model
model = Sequential()

# Load the model weights
model.load_weights("/content/model_weights.h5")

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
