import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Page configuration must be set before any Streamlit commands
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Function to load the model
@st.cache_resource
def load_fashion_model():
    model_path = 'best_fashion_cnn_model.keras'  # <-- changed to .keras
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        # Load model with compile=False to avoid metric issues with Keras 3+
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Tip: If this error persists, try saving the model again in the latest Keras format (.keras).")
        return None

# Function to preprocess the image and make predictions
def import_and_predict(image_data, model):
    try:
        size = (28, 28)
        # Convert image to grayscale and resize
        image = ImageOps.grayscale(ImageOps.fit(image_data, size, Image.Resampling.LANCZOS))
        img = np.asarray(image)
        img = img / 255.0  # Normalize
        img_reshape = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Load the model once
model = load_fashion_model()
if model is None:
    st.stop()

# Streamlit UI Design
st.title("🧥 Fashion Dataset by Espiritu_Castillo")
st.write(
    """
    Welcome to the Fashion Item Classifier! 
    Upload an image of a fashion item, and the model will predict what type of item it is.
    """
)

st.sidebar.write("## Instructions")
st.sidebar.write(
    """
    1. Upload a photo of a fashion item (jpg or png).
    2. Wait for the model to process and predict.
    3. See the prediction result below the uploaded image.
    """
)

file = st.file_uploader("Choose a photo from your computer", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file to get started.")
else:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform prediction
    prediction = import_and_predict(image, model)

    if prediction is not None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        result_class = np.argmax(prediction)
        result_label = class_names[result_class]
        confidence = prediction[0][result_class]

        st.write("## Prediction Result")
        st.write(f"**Item:** {result_label}")
        st.write(f"**Confidence:** {confidence:.2%}")

        st.balloons()  # Add some celebratory balloons when a prediction is made
