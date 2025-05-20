import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model('best_fashion_cnn_model.h5')

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Fashion Item Classifier")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    if st.button("Classify"):
        prediction = model.predict(image_array)
        class_id = np.argmax(prediction)
        st.success(f"Prediction: {class_names[class_id]} ({prediction[0][class_id]*100:.2f}%)")
