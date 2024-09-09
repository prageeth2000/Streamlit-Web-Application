
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained model (e.g., MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

def classify_image(img):
    predictions = model.predict(img)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)
    return decoded_predictions[0]

# Streamlit App
st.title("Image Classification App")
st.write("Upload an image and the app will classify it using a pre-trained MobileNetV2 model.")

# Sidebar
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = preprocess_image(image)

    # Progress bar
    st.sidebar.text("Classifying...")
    progress = st.sidebar.progress(0)

    for i in range(100):
        progress.progress(i + 1)

    # Classify the image
    predictions = classify_image(img)
    st.write("Classifications:")
    for i, pred in enumerate(predictions):
        st.write(f"{pred[1]}: {round(pred[2] * 100, 2)}%")

    # Displaying Graph
    st.subheader("Prediction Probabilities")
    labels = [pred[1] for pred in predictions]
    scores = [pred[2] for pred in predictions]
    plt.barh(labels, scores, color='skyblue')
    st.pyplot(plt)
