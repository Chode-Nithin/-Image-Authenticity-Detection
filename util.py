
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st

model = None
labels = ['real', 'fake']
st.title("Authenticity_Detection")

st.wirte("Upload the picture here!")

def load_model():
    global model
    model = tf.keras.models.load_model('fakevsreal_weights.h5')
file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])

def classify_image(file_uploaded):
    if model is None:
        load_model()

    image = Image.open(file_uploaded) # reading the image
    image = image.resize((128, 128)) # resizing the image to fit the trained model
    image = image.convert("RGB") # converting the image to RGB
    img = np.asarray(image) # converting it to numpy array
    img = np.expand_dims(img, 0)
    predictions = model.predict(img) # predicting the label
    label = labels[np.argmax(predictions[0])] # extracting the label with maximum probablity
    probab = float(round(predictions[0][np.argmax(predictions[0])]*100, 2))

    result = {
        'label': label,
        'probablity': probab
    }

    return result
rs =classify_image(file_uploaded)
st.markdown(rs)