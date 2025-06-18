import gradio as gr
import pytesseract
import cv2
import numpy as np
from PIL import Image

def ocr_from_image(image):
    # Convert PIL Image to OpenCV
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

iface = gr.Interface(
    fn=ocr_from_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="OCR Prescription Detection",
    description="Upload a prescription image and get extracted text using OCR"
)

if __name__ == "__main__":
    iface.launch()
