import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os
import h5py

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
    model_path = 'best_fashion_cnn_model.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    # Create a fresh model with the same architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    try:
        # Compile the model (needed for prediction even if we load weights)
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        # Attempt to load weights directly
        try:
            # Use h5py to extract weights directly
            with h5py.File(model_path, 'r') as f:
                # Check if it's a model or just weights
                if 'model_weights' in f:
                    # This is a full model file, extract weights
                    weight_names = [n.decode('utf8') for n in f.attrs['layer_names']]
                    model.build((None, 28, 28, 1))  # Build model to create weight variables
                    model.load_weights(model_path)
                else:
                    # This is just a weights file
                    model.load_weights(model_path)
                
            st.success("Successfully loaded model weights")
            return model
        except Exception as e:
            st.error(f"Failed to load weights: {e}")
            
            # One last attempt - direct model loading with custom_objects
            try:
                model = tf.keras.models.load_model(
                    model_path, 
                    compile=False,
                    custom_objects={'batch_shape': None}
                )
                st.success("Successfully loaded model with custom_objects")
                return model
            except Exception as e2:
                st.error(f"Also failed with custom_objects: {e2}")
                
                # Display a warning that predictions will be random
                st.warning("⚠️ Using untrained model. Predictions will be random!")
                return model
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess and predict function
def import_and_predict(image_data, model):
    try:
        size = (28, 28)
        # Convert to grayscale and resize
        image = ImageOps.grayscale(ImageOps.fit(image_data, size, Image.Resampling.LANCZOS))
        
        # Fashion MNIST has white items on black background
        # Check if we need to invert colors - if the image is predominantly dark (like a photo)
        img_array = np.array(image)
        mean_pixel = np.mean(img_array)
        if mean_pixel > 128:  # If image is predominantly bright
            image = ImageOps.invert(image)  # Invert to match Fashion MNIST
        
        # Convert to numpy array and normalize exactly as in training
        img = np.array(image).astype('float32') / 255.0
        
        # Show processed image for debugging
        st.write("Processed image (how the model sees it):")
        st.image(img, width=150)
        
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
