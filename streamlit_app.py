import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Page config
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load model function
@st.cache_resource
def load_fashion_model():
    model_path = 'best_fashion_cnn_model.h5'  # or .h5 if you keep h5 format
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess and predict function
def import_and_predict(image_data, model):
    try:
        size = (28, 28)
        # Convert to grayscale, resize and invert colors if needed (Fashion MNIST is white on black)
        image = ImageOps.grayscale(ImageOps.fit(image_data, size, Image.Resampling.LANCZOS))
        
        # Convert to numpy array
        img = np.array(image).astype('float32') / 255.0  # Normalize to 0-1
        
        # Reshape to (1, 28, 28, 1)
        img_reshape = img.reshape(1, 28, 28, 1)
        
        # Predict
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Load model
model = load_fashion_model()
if model is None:
    st.stop()

# UI
st.title("🧥 Fashion Clothes Classifier")
st.write(
    """
    Upload a fashion item photo (grayscale or color) and the model will predict its class.
    """
)

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image (jpg or png).
2. The model expects a 28x28 grayscale image.
3. Wait for prediction and see results below.
""")

file = st.file_uploader("Upload Image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file to get started.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = import_and_predict(image, model)
    if prediction is not None:
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
        ]
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        st.subheader("Prediction Result")
        st.write(f"**Item:** {class_names[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2%}")

        st.balloons()
