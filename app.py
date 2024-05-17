# Save this as `app.py`
import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Streamlit app
st.title("fashion_mnist")

# Load the model architecture from JSON
with open("/content/model.json") as json_file:
    loaded_model_json = json_file.read()

# Load the model weights
model.load_weights("/content/model.json")

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
