🧾 OCR Prescription Detection
A deep learning–based Flask web app for detecting and extracting text from prescription images using a CNN + LSTM + BiLSTM OCR model. Supports both printed and handwritten prescriptions with advanced image preprocessing.

📄 Paper: Presented at IEEE CCPIS 2025 | DOI: 10.1109/CCPIS65231.2025.11234194

📌 Features
Upload prescription images (JPG, PNG, PDF)

Custom CNN + LSTM + BiLSTM model with attention mechanism

Advanced preprocessing (adaptive thresholding, skew correction, noise reduction)

Optional Tesseract and TrOCR support for comparison

Medical term extraction with regex and spell correction

Export results as JSON or CSV

Simple Flask web interface

🏗️ Architecture
Input → Preprocessing → CNN Encoder → BiLSTM Decoder → Attention → Output

Component	Function
CNN Encoder	Extracts spatial features
BiLSTM	Captures sequential dependencies
Attention	Focuses on relevant regions
ViT (optional)	Medicine name classification
📊 Performance
Model	Existing	Proposed
ViT	90.00%	91.00%
RNN-LSTM	86.00%	89.84%
YOLO	82.89%	85.50%
CNN	70.45%	76.56%
🚀 Quick Start
bash
git clone https://github.com/Dutta-Raj/OCR-Prescription-Detection.git
cd OCR-Prescription-Detection
pip install -r requirements.txt
python app.py
Then open http://localhost:5000 in your browser.

📁 Project Structure
text
OCR-Prescription-Detection/
├── app.py                  # Flask web application
├── train.py                # Model training script
├── ocr prescription detection.ipynb
├── requirements.txt
├── Data 1/                 # Training dataset
├── Data 2/                 # Validation & testing dataset
└── LICENSE
🧠 Methodology
Preprocessing - PDF to image, grayscale, adaptive thresholding, resizing (128x32), normalization, skew correction, noise removal, CLAHE, data augmentation

Model - CNN encoder extracts features → BiLSTM decoder processes sequences → Attention mechanism focuses on characters

Post-processing - Spell checking, regex filtering (dates, drug names), fuzzy matching for typos

📈 Training Results (20 epochs)
Metric	Final Value
Box Loss	1.066
Class Loss	1.442
DFL Loss	1.093
GPU Memory	~12GB
🔗 Citation
bibtex
@inproceedings{das2025deep,
  title={A Deep Learning-Driven OCR Framework for Medical Prescription Analysis},
  author={Das, Riddhi Pratim and Dey, Raghunath and Mondal, Raktim and Gangopadhyay, Rhitav and Dutta, Rajdeep and Piri, Jayashree},
  booktitle={2025 2nd International Conference on CCPIS},
 year={2025},
  doi={10.1109/CCPIS65231.2025.11234194}
}
📝 License
MIT License - see LICENSE file
