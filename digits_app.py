# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:01:35 2022

@author: Sony
"""
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import cv2
from PIL import Image,ImageEnhance
import warnings
warnings.filterwarnings('ignore')
model=load_model('model_trained.p')
st.title("Digits Digits Classification Web App")
st.subheader('Kindly upload your file here')

image_file=st.file_uploader(('Upload Image'), type=['png','jpg','jpeg'])
if image_file is not None:
    input_image=Image.open(image_file)
    st.image(input_image)

if st.button('Predict the Image'):
    input_image=np.asarray(input_image)
    grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image_resize = cv2.resize(grayscale, (28, 28))
    input_image_resize = input_image_resize/255
    image_reshaped = np.reshape(input_image_resize, [1,28,28,1])
    input_prediction = model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    st.header('The Handwritten Digit is recognised as :', input_pred_label)
    st.header(input_pred_label)
    
   
    
    

