from flask import Flask, request, render_template_string, jsonify
import pytesseract
from PIL import Image
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = Flask(__name__)

# Load TrOCR once
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


# ---------------- OCR Functions ----------------
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )
    return thresh

def run_tesseract(image_path):
    thresh = preprocess_image(image_path)
    text = pytesseract.image_to_string(Image.fromarray(thresh), config="--oem 1 --psm 6")
    return text.strip()

def run_trocr(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


# ---------------- HTML Template ----------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>OCR Prescription Detection</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #eef2f7;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .card {
      background: #fff;
      padding: 30px;
      border-radius: 15px;
      width: 600px;
      box-shadow: 0px 8px 25px rgba(0,0,0,0.1);
      text-align: center;
      transition: 0.3s ease;
    }
    .card:hover {
      box-shadow: 0px 12px 30px rgba(0,0,0,0.15);
    }
    h2 {
      color: #2c3e50;
      margin-bottom: 20px;
    }
    input[type="file"], select {
      padding: 12px;
      font-size: 15px;
      border-radius: 8px;
      border: 1px solid #ccc;
      margin-top: 10px;
      width: 80%;
    }
    button {
      padding: 12px 25px;
      margin-top: 15px;
      font-size: 16px;
      font-weight: bold;
      background: #3498db;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: 0.3s ease;
    }
    button:hover {
      background: #2980b9;
    }
    .output {
      margin-top: 25px;
      text-align: left;
      padding: 15px;
      background: #f9f9f9;
      border-radius: 10px;
      border-left: 5px solid #3498db;
      max-height: 250px;
      overflow-y: auto;
      font-family: monospace;
      white-space: pre-wrap;
    }
    .image-preview {
      margin-top: 20px;
    }
    img {
      max-width: 100%%;
      border-radius: 10px;
      margin-top: 10px;
      box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
    }
  </style>
</head>
<body>
  <div class="card">
    <h2>🩺 OCR Prescription Detection</h2>
    <form method="POST" action="/ocr" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" onchange="previewImage(event)" required>
      <br>
      <label for="engine"><b>Select OCR Engine:</b></label>
      <select name="engine">
        <option value="tesseract">Tesseract (Printed)</option>
        <option value="trocr">TrOCR (Handwriting)</option>
      </select>
      <br>
      <button type="submit">Upload & Detect</button>
    </form>

    <div class="image-preview" id="preview"></div>

    {% if extracted_text %}
      <div class="output">
        <h3>📜 Extracted Text:</h3>
        <p>{{ extracted_text }}</p>
      </div>
    {% endif %}
  </div>

  <script>
    function previewImage(event) {
      var preview = document.getElementById('preview');
      preview.innerHTML = "";
      var file = event.target.files[0];
      if(file) {
        var reader = new FileReader();
        reader.onload = function(e) {
          var img = document.createElement("img");
          img.src = e.target.result;
          preview.appendChild(img);
        }
        reader.readAsDataURL(file);
      }
    }
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    engine = request.form.get("engine", "tesseract")
    image_path = "uploaded.png"
    file.save(image_path)

    text = run_tesseract(image_path) if engine == "tesseract" else run_trocr(image_path)

    return render_template_string(HTML_TEMPLATE, extracted_text=text)

if __name__ == "__main__":
    app.run(debug=True)
