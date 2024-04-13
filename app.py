import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test', type=['png', 'jpg', 'jpeg','tif'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        return image
    return None

def predict(image):
    d = {}
    model = YOLO("best.pt")
    results = model.predict(image)[0]
    probs = results.probs
    
    d['GNB'] = probs.data[0]
    d['GNC'] = probs.data[1]
    d['GPB'] = probs.data[2]
    d['GPC'] = probs.data[3]

    max_key = max(d, key=d.get)  # Get the key with the maximum value
    max_value = d[max_key]  # Get the maximum value

    return max_key, max_value

def main():
    st.title('Corneal Infection Image Classification')
    image = load_image()
    if image is not None:
        max_key, max_value = predict(image)
        st.write("Predicted Class:", max_key)
        st.write("Confidence:", round(np.array(max_value)*100,2),"%")

if __name__ == '__main__':
    main()
