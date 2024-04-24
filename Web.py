import streamlit as st 
import matplotlib.pyplot as plt
import cv2
import numpy as np
import joblib

import tensorflow as tf

model = tf.keras.models.load_model("model_directory")



st.title("CrophyPhi")


def predict(uploaded_file):
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (225, 225), interpolation=cv2.INTER_AREA)
        image = np.array(image)
        image = image / 255.
        image = image.reshape(-1,225,225,3)
        st.image(image)
        result = np.round(model.predict([image]))
        original_data = np.argmax(result, axis=1)
        print(original_data)
        data = ['Healthy','Rusty','Powdery']
        st.write(data[original_data[0]])
        


def main():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if st.button("Predict"):
        predict(uploaded_file)
    
    
if __name__ == "__main__":
    main()
