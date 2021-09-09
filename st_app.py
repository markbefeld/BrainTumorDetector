import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img

st.title('Brain Tumor Prediction')


page = st.sidebar.selectbox(
'Select a Page',('Tumor Detector')
)

if page == 'Tumor Detector':

    new_model = tf.keras.models.load_model('saved_model/my_model')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        new_size = (256,256)

        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        pred_array = []

        try:
            image = image.resize(new_size)
            pred_arr = img_to_array(image) / 255
            pred_array.append(pred_arr)
            X = np.array(pred_array)

            if new_model.predict_classes(X) == 1:
                st.write('This scan contains a brain tumor.')
            else:
                st.write('This scan does not contain a brain tumor.')
        except:
            st.write(f'Error for file')

    else:
        st.write('Please try another file.')
