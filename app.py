from flask import Flask, request, render_template_string, jsonify
import cv2
import pytesseract
import numpy as np
from PIL import Image

app = Flask(__name__)

# Simple HTML upload form
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>OCR Prescription Detection</title>
</head>
<body>
    <h2>Upload Prescription Image</h2>
    <form method="POST" action="/ocr" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Upload & Detect">
    </form>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Read image with PIL then convert to OpenCV format
    img = Image.open(file.stream).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Run OCR
    text = pytesseract.image_to_string(img_cv)

    return f"<h3>Detected Text:</h3><pre>{text}</pre><br><a href='/'>Go Back</a>"

if __name__ == "__main__":
    app.run(debug=True)
