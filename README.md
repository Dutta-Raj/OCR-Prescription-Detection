# OCR-Seq2Seq: Handwritten Word Recognition using CNN + BiLSTM

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Colab](https://img.shields.io/badge/Run%20on-Colab-green)

## 🧠 Project Title
**Handwritten OCR with CNN-BiLSTM-Seq2Seq Model**

## 📌 Description
This project implements an AI-based Optical Character Recognition (OCR) system using a CNN for feature extraction and a BiLSTM with a sequence-to-sequence architecture to decode handwritten words. The model is trained and evaluated using image-label pairs (folder-structured), with each image representing a word and folder name as the target text.

---

## 🗂️ Folder Structure
/data
└── word_label_1/
└── image1.png
└── image2.png
└── word_label_2/
└── image1.png

---

## 🚀 Features
- CNN-based image feature extraction
- BiLSTM encoder-decoder for sequence prediction
- Mixed precision training for improved speed
- Real-time performance visualization (accuracy, loss)
- F1 Score and character-level accuracy evaluation
- Auto brightness and contrast augmentation
- Colab compatible with Google Drive integration

---

## 📦 Requirements

```bash
pip install tensorflow matplotlib seaborn scikit-learn pillow
