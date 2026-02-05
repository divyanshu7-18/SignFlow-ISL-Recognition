# 🧠 SignFlow AI Project

### Indian Sign Language (ISL) Recognition Using Deep Learning

---

## 📌 Project Description

**SignFlow** is an AI-powered system that recognizes **Indian Sign Language (ISL)** gestures from video input and converts them into meaningful text. The project uses **computer vision**, **hand pose estimation**, and **deep learning** to enable real-time sign language recognition.

This system is designed to help reduce communication barriers for the **hearing and speech impaired community**, with a focus on **Indian Sign Language**, which is often underrepresented in existing research.

---

## 🎯 Objectives

* Recognize ISL gestures from recorded and live video
* Extract pose and hand landmarks from video frames
* Train deep learning models for gesture classification
* Enable real-time prediction using a webcam
* Build a modular and extensible AI pipeline

---

## 🚀 Key Features

* 📷 Real-time gesture recognition
* ✋ Hand & pose landmark extraction
* 🧠 Deep learning–based classification
* 📝 Gesture-to-text output
* 🔄 Train, evaluate, and test pipeline
* 🗂️ Clean and scalable project structure

---

## 🏗️ Project Architecture

```
Video Input (Live / Recorded)
        ↓
Frame Processing
        ↓
Pose & Hand Landmark Extraction
        ↓
Feature Normalization
        ↓
Deep Learning Model
        ↓
Gesture Prediction
        ↓
Text Output
```

---

## 📂 Project Structure

```
SIGNFLOW_AI_PROJECT/
│
├── .vscode/                 # VS Code configuration
│
├── checkpoints/             # Saved model checkpoints
│
├── data/
│   ├── raw/                 # Original datasets
│   │   ├── ISL_CSLTR/
│   │   └── Kaggle_Words/
│   │
│   ├── processed/           # Preprocessed data
│   │   ├── poses/           # Extracted landmark data
│   │   └── videos/          # Processed video files
│
├── dataset/                 # Final training/testing dataset
│
├── src/
│   ├── data/                # Data handling scripts
│   ├── model/               # Model architecture files
│   ├── __init__.py
│   ├── config.py            # Global configuration settings
│   ├── train.py             # Model training logic
│   ├── evaluate.py          # Model evaluation
│   ├── predict.py           # Offline prediction
│   ├── utils.py             # Utility functions
│
├── venv/                    # Python virtual environment
│
├── isl_model.h5             # Trained deep learning model
├── labels.npy               # Gesture label mappings
├── history.pkl              # Training history
│
├── predict_live.py           # Real-time webcam prediction
├── train_model.py            # Training entry script
├── test_dataset.py           # Dataset validation/testing
├── requirements.txt          # Project dependencies
└── README.md
```

---

## 📊 Dataset Information

* **Raw Datasets**

  * ISL_CSLTR
  * Kaggle_Words
* Data consists of:

  * Videos of ISL gestures
  * Multiple samples per gesture
* Preprocessing includes:

  * Frame extraction
  * Hand & pose landmark detection
  * Normalization and labeling

---

## 🧠 Model Details

* **Input:** Hand and pose landmark coordinates (x, y, z)
* **Architecture:**

  * CNN for spatial feature extraction
  * LSTM for temporal gesture modeling
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam
* **Output:** Predicted ISL gesture class

---

## ⚙️ Installation & Setup

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### 🔹 Train the Model

```bash
python train_model.py
```

### 🔹 Evaluate the Model

```bash
python src/evaluate.py
```

### 🔹 Test Dataset Integrity

```bash
python test_dataset.py
```

### 🔹 Predict from Saved Data

```bash
python src/predict.py
```

### 🔹 Real-Time Gesture Recognition

```bash
python predict_live.py
```

---

## 📈 Results

* Accurate recognition of trained ISL gestures
* Low-latency real-time predictions
* Stable performance with sufficient lighting and clear gestures

---

## ⚠️ Limitations

* Limited gesture vocabulary (dataset dependent)
* Performance affected by lighting conditions
* Complex sentence-level recognition not implemented
* Occlusion and overlapping hands reduce accuracy

---

## 🔮 Future Improvements

* 🔊 Text-to-Speech integration
* 🧾 Sentence-level gesture recognition
* 📱 Mobile and web deployment
* ☁️ Cloud-based inference
* 🧠 Transformer-based temporal models

---

## 🎓 Academic Use

This project is suitable for:

* AI / ML / Deep Learning coursework
* Final-year engineering projects
* Assistive technology research
* Computer vision applications

---

## 👨‍💻 Author

**Divyanshu Kumar**
Artificial Intelligence & Machine Learning
Project: *SignFlow – Indian Sign Language Recognition*

---

## 📜 License

This project is intended for **educational and research purposes only**.


