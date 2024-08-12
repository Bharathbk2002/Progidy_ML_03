import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

model=load_model(r'C:\Users\bhara\Internship_1\image_classify.keras')
data_cat=['Cat', 'Dog']

img_height=180
img_width=180

st.markdown("<h2><b>Enter Image Name</b></h2>", unsafe_allow_html=True)
image = st.text_input('', 'dog.jpg')
image_load=tf.keras.utils.load_img(image,target_size=(img_height,img_width))
img_arr=tf.keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict=model.predict(img_bat)

score=tf.nn.softmax(predict)
print(score)
st.image(image)
st.markdown(f"**<span style='font-size:24px;'>Cat/Dog in image is<span style='color:red;'> {data_cat[np.argmax(score)]}</span>**", unsafe_allow_html=True)
st.markdown(
    f"**<span style='font-size:24px;'>with accuracy of <span style='color:red;'>{np.max(score) * 100:.2f}%</span>**", 
    unsafe_allow_html=True
)
