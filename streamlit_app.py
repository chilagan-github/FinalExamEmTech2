import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load trained CNN model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("best_fashion_cnn_model.h5")

model = load_cnn_model()

# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# App title
st.title("👕🩳 Fashion Item Classifier (CNN)")

# Upload image
uploaded_file = st.file_uploader("Upload a clothing item image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image_resized = image.resize((28, 28))
    
    st.image(image_resized, caption="Uploaded Image (processed)", use_column_width=True)

    # Preprocess
    img_array = np.array(image_resized).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"🧠 Predicted Class: **{predicted_class}**")
    st.info(f"📊 Confidence: {confidence:.2f}%")
