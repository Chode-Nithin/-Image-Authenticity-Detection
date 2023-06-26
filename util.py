import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

labels = ['real', 'fake']
probab = 0

st.title("Authenticity_Detection")
st.write("Upload the picture here!")

@st.cache_resource()
def get_model():
    model_path = 'fakevsreal_weights.h5'
    model = tf.keras.models.load_model(model_path)
    return model

file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])

def classify_image(file_uploaded):
    if file_uploaded is not None:
        model = get_model()

        image = Image.open(file_uploaded)  # reading the image
        image = image.resize((128, 128))  # resizing the image to fit the trained model
        image = image.convert("RGB")  # converting the image to RGB
        img = np.asarray(image)  # converting it to numpy array
        img = np.expand_dims(img, 0)

        # Preprocess the image using ResNet50 preprocessing
        img = preprocess_input(img)

        # Create an instance of the ResNet50 model
        resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        resnet_model.trainable = False

        # Pass the input through the ResNet50 model
        resnet_output = resnet_model(img)

        # Flatten the output tensor
        flattened_output = tf.keras.layers.Flatten()(resnet_output)

        # Reshape the flattened output to match the expected input shape of the sequential model
        reshaped_output = tf.keras.layers.Reshape((4, 4, 2048))(flattened_output)

        # Pass the reshaped output through your existing model's layers
        predictions = model(reshaped_output)

        # Extract the label with maximum probability
        label = labels[np.argmax(predictions[0])]
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




      
     



# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import streamlit as st

# labels = ['real', 'fake']
# probab=0
# st.title("Authenticity_Detection")
# st.write("Upload the picture here!")

# @st.cache_resource()
# def get_model():
#     model_path ='fakevsreal_weights.h5'
#     model = tf.keras.models.load_model(model_path)
#     return model

# file_uploaded = st.file_uploader("Choose the Image File", type=["jpg", "png", "jpeg"])

# def classify_image(file_uploaded):
#     if file_uploaded is not None:
#         model = get_model()

#         image = Image.open(file_uploaded) # reading the image
#         image = image.resize((128, 128)) # resizing the image to fit the trained model
#         image = image.convert("RGB") # converting the image to RGB
#         img = np.asarray(image) # converting it to numpy array
#         img = np.expand_dims(img, 0)
#         predictions = model.predict(img) # predicting the label
#         label = labels[np.argmax(predictions[0])] # extracting the label with maximum probability
#         probab = float(predictions[0][np.argmax(predictions[0])])

#         result = {
#             'label': label,
#             'probability': probab
#         }

#         return result
#     else:
#         return None

# rs = classify_image(file_uploaded)

# if rs is not None:
#     st.write("Your Image is:", rs['label'])
#     st.write("Probability:", rs['probability'])


    
