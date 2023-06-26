
import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st

labels = ['real', 'fake']
probab=0
st.title("Authenticity_Detection")
st.write("Upload the picture here!")

@st.cache_resource()
def get_model():
    model_path ='fakevsreal_weights.h5'
    model = tf.keras.models.load_model(model_path)
    return model

file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])

def classify_image(file_uploaded):
    if file_uploaded is not None:
        model = get_model()

        image = Image.open(file_uploaded) # reading the image
        image = image.resize((128, 128)) # resizing the image to fit the trained model
        image = image.convert("RGB") # converting the image to RGB
        img = np.asarray(image) # converting it to numpy array
        img = np.expand_dims(img, 0)
        predictions = model.predict(img) # predicting the label
        label = labels[np.argmax(predictions[0])] # extracting the label with maximum probability
        probab = float(predictions[0][np.argmax(predictions[0])])

        result = {
            'label': label,
            'probability': probab
        }

        return result
    else:
        return None

rs = classify_image(file_uploaded)

if rs is not None:
    st.write("Your Image is:", rs['label'])
    st.write("Probability:", rs['probability'])


    
