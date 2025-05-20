import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Page configuration
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_fashion_model():
    model_path = "best_fashion_cnn_model(1).h5"  # folder, not .h5
    if not os.path.exists(model_path):
        st.error(f"Model folder not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def import_and_predict(image_data, model):
    try:
        size = (28, 28)
        # Convert to grayscale and resize
        image = ImageOps.grayscale(ImageOps.fit(image_data, size, Image.Resampling.LANCZOS))
        img = np.array(image).astype('float32') / 255.0  # normalize like training
        img = img.reshape(1, 28, 28, 1)  # add batch and channel dims
        prediction = model.predict(img)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

model = load_fashion_model()
if model is None:
    st.stop()

st.title("🧥 Fashion Classifier")
st.write(
    """
    Upload a fashion item photo, and the model will predict the clothing type.
    """
)

st.sidebar.write("## Instructions")
st.sidebar.write("""
1. Upload a jpg or png image.
2. The model expects a 28x28 grayscale image.
3. Wait for prediction results below.
""")

file = st.file_uploader("Upload Image", type=["jpg", "png"])
if file is None:
    st.text("Please upload an image file to get started.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = import_and_predict(image, model)

    if prediction is not None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        st.write("## Prediction Result")
        st.write(f"**Item:** {class_names[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.balloons()
