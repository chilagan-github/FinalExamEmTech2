import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="Fashion Classifier", page_icon="👗")

@st.cache_resource
def load_fashion_model():
    model_path = 'best_fashion_cnn_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
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
        image = ImageOps.grayscale(ImageOps.fit(image_data, size, Image.Resampling.LANCZOS))
        img_array = np.array(image)
        mean_pixel = np.mean(img_array)
        if mean_pixel > 128:
            image = ImageOps.invert(image)
        img = np.array(image).astype('float32') / 255.0
        st.write("Processed image:")
        st.image(img, width=150)
        img_reshape = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img_reshape)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

model = load_fashion_model()
if model is None:
    st.stop()

st.title("🧥 Fashion Clothes Classifier")
st.write("Upload a fashion item photo and get a prediction.")

file = st.file_uploader("Upload Image", type=["jpg", "png"])
if file is not None:
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
else:
    st.text("Please upload an image file to get started.")
