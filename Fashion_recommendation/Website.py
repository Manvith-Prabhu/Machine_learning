import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 
import numpy as np
import os
import pickle
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import cv2

model= ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model=tf.keras.Sequential([
model,
GlobalMaxPooling2D()
])

feature_list=np.array(pickle.load(open('D:\Programs\Python\ML_notebooks\Fashion_recommendation/feature_list_embeddings.pkl','rb')))
filenames=pickle.load(open('D:\Programs\Python\ML_notebooks\Fashion_recommendation/filenames.pkl','rb'))

st.title("Fashion Recommender System")

def save_file(uploaded_file):
    try:
        with open(os.path.join("D:\Programs\Python\ML_notebooks\Fashion_recommendation/Uploaded_image",uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img_array=np.expand_dims(img_array,axis=0)
    preprocessed_img=preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    norm_result=result/np.linalg.norm(result)
    return norm_result

def recommend(features,feature_list):
    neighbours=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
    neighbours.fit(feature_list)
    distance, indices=neighbours.kneighbors([features])
    return indices



uploaded_file=st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_file(uploaded_file):
        display_image=Image.open(uploaded_file)
        st.image(display_image)
        features=feature_extraction(os.path.join("D:\Programs\Python\ML_notebooks\Fashion_recommendation/Uploaded_image",uploaded_file.name),model)
        indices=recommend(features,feature_list)
        col1,col2,col3,col4,col5=st.columns(5)
        
        with col1:
            image = plt.imread(filenames[indices[0][0]])
            new_image = cv2.resize(image, (1000, 1000))
            st.image(new_image)

        with col2:
            image = plt.imread(filenames[indices[0][1]])
            new_image = cv2.resize(image, (1000, 1000))
            st.image(new_image)

        with col3:
            image = plt.imread(filenames[indices[0][2]])
            new_image = cv2.resize(image, (1000, 1000))
            st.image(new_image)

        with col4:
            image = plt.imread(filenames[indices[0][3]])
            new_image = cv2.resize(image, (1000, 1000))
            st.image(new_image)

        with col5:
            image = plt.imread(filenames[indices[0][4]])
            new_image = cv2.resize(image, (1000, 1000))
            st.image(new_image)

    else:
        st.header("Some error occured in file upload")


