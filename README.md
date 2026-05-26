<div align="center">

# 🩺 Skin Disease Detection using Deep Learning & Multi-Modal Fusion

### AI-Powered Multi-Class Skin Lesion Classification using Dermoscopic Images and Patient Metadata

<p align="center">
  <a href="https://github.com/Joshinx17/skin-disease-detection">
    <img src="https://img.shields.io/github/stars/Joshinx17/skin-disease-detection?style=for-the-badge" />
  </a>
  <a href="https://github.com/Joshinx17/skin-disease-detection/network/members">
    <img src="https://img.shields.io/github/forks/Joshinx17/skin-disease-detection?style=for-the-badge" />
  </a>
  <a href="https://github.com/Joshinx17/skin-disease-detection/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Joshinx17/skin-disease-detection?style=for-the-badge" />
  </a>
  <a href="https://skin-disease-classification-project.streamlit.app/">
    <img src="https://img.shields.io/badge/Live-Demo-red?style=for-the-badge&logo=streamlit" />
  </a>
</p>

<p align="center">
  <a href="https://joshinx17.github.io/skin-disease-detection/">🌐 Project Website</a>
  •
  <a href="https://skin-disease-classification-project.streamlit.app/">🚀 Streamlit App</a>
  •
  <a href="https://ieeexplore.ieee.org/document/11465623">📄 Research Paper</a>
</p>

</div>

---

# 📌 Overview

Skin diseases remain one of the most challenging medical conditions to diagnose accurately due to high visual similarity among lesions, class imbalance, and dependency on expert dermatological analysis.

This project presents a **Deep Learning based Multi-Class Skin Disease Detection System** that combines:

- 🧠 **Dermoscopic Image Analysis**
- 📋 **Patient Metadata Fusion**
- 🔬 **Transfer Learning**
- ⚡ **Real-Time Prediction Interface**

The system uses a **Fusion Architecture** that integrates a **ResNet50 CNN image branch** with a **metadata processing network** to improve diagnostic accuracy and clinical reliability.

The model is capable of classifying **7 different skin lesion categories** using dermoscopic images from the **HAM10000 dataset**.

---

# ✨ Key Features

✅ Multi-class skin disease classification  
✅ Fusion model combining image + metadata  
✅ Transfer Learning using ResNet50  
✅ Streamlit-based web application  
✅ Research-backed implementation  
✅ Clinically sensitivity-oriented design  
✅ Real-time lesion analysis  
✅ Modern responsive deployment  
✅ ROC-AUC up to **1.00**  
✅ Melanoma Recall of **99%**

---

# 🧠 Supported Disease Classes

| Class Code | Disease Name |
|---|---|
| `akiec` | Actinic Keratoses |
| `bcc` | Basal Cell Carcinoma |
| `bkl` | Benign Keratosis-like Lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `vasc` | Vascular Lesions |

---

# 🏗️ System Architecture

## 🔹 Fusion Model Architecture

```text
                ┌────────────────────┐
                │ Dermoscopic Image  │
                └─────────┬──────────┘
                          │
                    ResNet50 CNN
                          │
                  Deep Image Features
                          │
                          ├──────────────┐
                          │              │
                          │      Patient Metadata
                          │      (Age, Sex, Localization)
                          │              │
                          │       Metadata Network
                          │              │
                          └──────┬───────┘
                                 │
                         Feature Fusion
                                 │
                       Fully Connected Layers
                                 │
                         Softmax Classification
                                 │
                    Final Disease Prediction
```

---

# 📊 Model Performance

## 📈 Evaluation Results

| Metric | Score |
|---|---|
| Overall Accuracy | **89%** |
| Weighted F1-Score | **0.89** |
| Weighted Precision | **0.93** |
| Melanoma Recall | **0.99** |
| ROC-AUC Range | **0.98 – 1.00** |

---

# 📉 Classification Report

| Disease Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Benign Keratosis | 0.89 | 0.91 | 0.90 |
| Melanocytic Nevi | 1.00 | 0.85 | 0.92 |
| Dermatofibroma | 1.00 | 1.00 | 1.00 |
| Melanoma | 0.56 | 0.99 | 0.71 |
| Vascular Lesions | 0.88 | 1.00 | 0.93 |
| Basal Cell Carcinoma | 0.86 | 1.00 | 0.92 |
| Actinic Keratosis | 0.93 | 1.00 | 0.96 |

---

# 🖼️ Demo & Screenshots

## 🌐 Website Preview

> Add website screenshots here

```md
/assets/website-preview.png
```

---

## 🚀 Streamlit Application

> Add Streamlit app screenshots here

```md
/assets/streamlit-demo.png
```

---

## 📊 Training Curves

> Add training accuracy/loss graphs here

```md
/assets/training-curves.png
```

---

## 🔍 Prediction Output

> Add prediction interface screenshots/GIFs here

```md
/assets/prediction-demo.gif
```

---

# 🔬 Research Paper

## 📄 Published Research

**Title:**  
*A Deep Learning Framework for Automated Multi-Class Skin Disease Detection Using Dermoscopic Images*

### 👨‍💻 Authors
- Joshin K Saju
- Deepayan Thakur
- Parth Sharma
- Dr. Anand Pandey
- Dr. Mohd Tajammul

### 📚 Publication
IEEE Conference Publication

### 🔗 Paper Link
https://ieeexplore.ieee.org/document/11465623

---

# 🧪 Dataset

This project uses the **HAM10000 Dataset**, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

### Dataset Characteristics

- 7 Skin Disease Classes
- Dermoscopic Images
- Real-world clinical imbalance
- Multi-class classification setup

---

# ⚙️ Tech Stack

## 🔹 Languages & Frameworks

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas
- PIL

## 🔹 Deep Learning

- ResNet50
- Transfer Learning
- Feature Fusion
- Softmax Classification

## 🔹 Deployment

- Streamlit Cloud
- GitHub Pages

---

# 🚀 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Joshinx17/skin-disease-detection.git
cd skin-disease-detection
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 🖥️ Usage

1. Upload a dermoscopic skin lesion image  
2. Enter patient metadata:
   - Age
   - Sex
   - Lesion Localization
3. Click **Analyze Lesion**
4. View:
   - Predicted disease
   - Confidence score
   - Probability distribution chart

---

# 📂 Project Structure

```bash
skin-disease-detection/
│
├── app.py
├── model.py
├── requirements.txt
├── README.md
├── LICENSE
├── assets/
│   ├── website-preview.png
│   ├── streamlit-demo.png
│   ├── prediction-demo.gif
│   └── training-curves.png
│
├── notebooks/
│   └── skindd-03.ipynb
│
├── models/
│   └── best_fusion_model_finetuned.pth
│
└── dataset/
```

---

# 🧠 Methodology

The proposed framework follows a multi-stage deep learning pipeline:

- Image preprocessing and normalization
- Transfer learning using ResNet50
- Metadata encoding and fusion
- Feature concatenation
- Fully connected classification layers
- Softmax-based multi-class prediction

The model was specifically optimized to minimize false negatives for malignant classes, improving clinical reliability.

---

# 📌 Future Improvements

- Explainable AI integration (Grad-CAM)
- Mobile application deployment
- Real-time clinical testing
- Improved dataset diversity
- Transformer-based hybrid fusion
- Lightweight edge deployment optimization

---

# 🤝 Contributors

| Name | Role |
|---|---|
| Joshin K Saju | Deep Learning & System Development |
| Deepayan Thakur | Research & Implementation |
| Parth Sharma | Data Processing & Analysis |

---

# 📜 License

This project is licensed under the **MIT License**.

You are free to:
- Use
- Modify
- Distribute
- Fork

with proper attribution.

---

# ⭐ Acknowledgements

- HAM10000 Dataset
- PyTorch
- Streamlit
- IEEE
- Open-source AI community

---

# 📬 Contact

## 👨‍💻 Joshin K Saju

📧 Email: joshinkoshys@gmail.com

🔗 GitHub:  
https://github.com/Joshinx17

---

<div align="center">

### ⭐ If you found this project useful, consider starring the repository!

</div>