import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Dummy OCR function (replace with your real model prediction)
def predict_text_from_image(image):
    # Convert PIL image to array
    image_array = np.array(image)
    
    # Optional: preprocessing (resize, grayscale, etc.)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Dummy return (replace with real model inference)
    return "Predicted text goes here."

# Gradio interface
iface = gr.Interface(
    fn=predict_text_from_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="OCR Prescription Detection",
    description="Upload a medical prescription image and get the extracted text using a deep learning model."
)

if __name__ == "__main__":
    iface.launch()
