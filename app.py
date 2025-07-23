# streamlit app example
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("mnist_model.h5")


st.title("MNIST Digit Classifier")
uploaded_file = st.file_uploader("Upload a 28x28 image")

if uploaded_file:
    img = Image.open(uploaded_file).convert('L').resize((28,28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 784)
    pred = np.argmax(model.predict(img_array), axis=1)
    st.image(img, caption=f"Prediction: {pred[0]}")