import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="expanded",
)

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
        
def import_and_predict(image_data, model):
    try:
        size = (28, 28)
        # Convert image to grayscale and resize
        image = ImageOps.fit(image_data.convert("L"), size, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [0,1]
        img = np.array(image).astype('float32') / 255.0

        # Uncomment if your input image background/foreground needs inversion
        img = 1 - img

        # Show preprocessed grayscale image for verification
        st.image(img, caption="Preprocessed 28x28 Grayscale Image", clamp=True)

        # Reshape for model input
        img = img.reshape(1, 28, 28, 1)

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
