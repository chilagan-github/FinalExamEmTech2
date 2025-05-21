import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load model with cached resource
@st.cache_resource
def load_fashion_model():
    weights_path = "best_fashion_cnn_model_no_mlp(1).h5"

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        GlobalAveragePooling2D(),
        
        Dense(10, activation='softmax')
    ])

    if not os.path.exists(weights_path):
        st.error(f"Weights file not found: {weights_path}")
        return None

    try:
        model.load_weights(weights_path)
        return model
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None

# Image preprocessing and prediction
def import_and_predict(image_data, model, invert=False):
    try:
        size = (28, 28)
        # Convert image to grayscale and resize
        image = ImageOps.fit(image_data.convert("L"), size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0,1]
        img = np.array(image).astype('float32') / 255.0

        if invert:
            img = 1 - img  # Optionally invert

        st.image(img, caption="Preprocessed 28x28 Grayscale Image", clamp=True)

        img = img.reshape(1, 28, 28, 1)  # Reshape for model input

        prediction = model.predict(img)
        return prediction
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Load model
model = load_fashion_model()
if model is None:
    st.stop()

# UI
st.title("🧥 Fashion Classifier")
st.write("""
Upload a fashion item photo, and the model will predict the clothing type.
""")

st.sidebar.write("## Instructions")
st.sidebar.write("""
1. Upload a jpg or png image.
2. The model expects a 28x28 grayscale image.
3. Optionally toggle image color inversion.
""")

file = st.file_uploader("Upload Image", type=["jpg", "png"])
if file is None:
    st.text("Please upload an image file to get started.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    invert_option = st.checkbox("Invert Image Colors (white on black)", value=True)

    prediction = import_and_predict(image, model, invert=invert_option)

    if prediction is not None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

        top_3 = prediction[0].argsort()[-3:][::-1]
        st.write("## Top Predictions")
        for i in top_3:
            st.write(f"**{class_names[i]}** — Confidence: {prediction[0][i]:.2%}")

        # Bar chart
        top_3_dict = {class_names[i]: float(prediction[0][i]) for i in top_3}
        st.bar_chart(top_3_dict)

        st.balloons()
