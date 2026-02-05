import streamlit as st
from keras.applications import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import io

# Page config
st.set_page_config(page_title="Bildklassificering", layout="centered")

# Title
st.title("Bildklassificering")
st.markdown("Ladda upp en bild för att klassificera den")

# Load model (cached)
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

# Predict function
def predict_image(img_array):
    """Predict image class"""
    # Resize to 224x224
    img_resized = Image.fromarray(img_array).resize((224, 224))
    x = np.array(img_resized)
    
    # Preprocess
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Predict
    model = load_model()
    preds = model.predict(x, verbose=0)
    
    # Decode predictions
    predictions = decode_predictions(preds, top=3)[0]
    return predictions

# Main app
uploaded_file = st.file_uploader("Välj en bild", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    img = Image.open(uploaded_file)
    st.image(img, width=600)
    
    # Predict
    with st.spinner("Klassificerar bild..."):
        img_array = np.array(img.convert('RGB'))
        predictions = predict_image(img_array)
    
    # Display results
    st.markdown("---")
    st.subheader("Resultat")
    
    for label, class_name, probability in predictions:
        percentage = probability * 100
        st.write(f"**{class_name}**: {percentage:.2f}%")
        st.progress(float(probability))
else:
    st.info("Ladda upp en bild för att börja")
