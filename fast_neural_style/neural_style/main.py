import streamlit as st
from PIL import Image
import pandas as pd
import io
from io import StringIO


import style
import os


st.title("PyTorch Style Transfer")

img = st.sidebar.selectbox(
    "Choose an image",
    (None, "amber.jpg", 'pizza.jpeg'),
    )

style_name = st.sidebar.selectbox(
    "Choose an image",
    ("candy",'mosaic', 'rain_princess', 'udnie')
    )

if img:
    input_image = "../images/content-images/" + img
    output_image = "../images/output-images/" + style_name + "-" + img
    image = Image.open(input_image)
    st.image(image, width=400)


st.write("### Or upload your own image")
uploaded_file = st.file_uploader(label="Upload an image")

if uploaded_file is not None:
    # Display the uploaded file
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption='Uploaded Image.', use_column_width=True, width=200)
    output_image = "../images/output-images/" + style_name + "-" + uploaded_file.name

model = "../saved_models/" + style_name + ".pth"

clicked = st.button("# Stylize Image")

if clicked:
    model = style.load_model(model)
    style.stylize(model, input_image, output_image)
    st.write("### Output Image:")
    styled_image = Image.open(output_image)
    st.image(styled_image, width=400)