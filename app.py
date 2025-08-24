from flask import Flask, request, render_template_string, jsonify
import cv2
import pytesseract
import numpy as np
from PIL import Image

app = Flask(__name__)

# Modern HTML template with Bootstrap
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OCR Prescription Detection</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(135deg, #f8f9fa, #e3f2fd);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .card {
      border-radius: 20px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .btn-custom {
      background: #007bff;
      color: white;
      border-radius: 10px;
      padding: 10px 20px;
      transition: 0.3s;
    }
    .btn-custom:hover {
      background: #0056b3;
    }
    .result-box {
      background: #f1f1f1;
      padding: 15px;
      border-radius: 10px;
      margin-top: 20px;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card p-4 text-center">
      <h2 class="mb-3">🧾 OCR Prescription Detection</h2>
      <form method="POST" action="/ocr" enctype="multipart/form-data">
        <input class="form-control mb-3" type="file" name="file" accept="image/*" required>
        <button type="submit" class="btn btn-custom">📤 Upload & Detect</button>
      </form>
      {% if text %}
        <div class="result-box text-start">
          <h5>✅ Detected Text:</h5>
          <pre>{{ text }}</pre>
        </div>
      {% endif %}
    </div>
  </div>
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

    return render_template_string(HTML_TEMPLATE, text=text)

if __name__ == "__main__":
    app.run(debug=True)
