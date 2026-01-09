## 👨‍💻 Authors

**Youssef Alouani**  
📧 youssef.alouani10@gmail.com
**Ouissam Aboujid**  
📧 aboujid.ouissam@gmail.com

---
# 🎭 Facial Emotion Recognition – Deep Learning Project

This project focuses on **facial emotion recognition** using Machine Learning and Deep Learning models (MLP, CNN, and Transfer Learning).  
It includes preprocessing, training, evaluation, and an interactive **Streamlit web app** for visualization and prediction.

---

## 📁 Project Structure

```
.
├── data/
│   ├── raw/                 
│   └── processed/          
│
├── experiments/
│   └── dl_checkpoints/      
│
├── notebooks/              
│
├── src/
│   ├── dl/                  
│   └── processing/         
│
├── streamlit_app/
│   ├── app.py               
│   └── modules/             
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / Mac**
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧹 Data Preprocessing

Run preprocessing pipeline:
```bash
python src/processing/preprocess.py
```

Or using Streamlit:
```bash
streamlit run streamlit_app/app.py
```
➡️ Navigate to **Processing → Run preprocessing**

---

## 🧠 Model Training

### MLP Baseline
```bash
python src/dl/mlp.py
```

### CNN Training
```bash
python src/dl/cnn.py
```

### Transfer Learning
```bash
python src/dl/transfer.py
```

Saved models:
```
experiments/dl_checkpoints/
```

---

## 📊 Evaluation

```bash
python src/dl/evaluation.py
```

Outputs:
- Accuracy & loss plots
- Training history (.json)

---

## 🌐 Run Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

Features:
- Dataset exploration
- Preprocessing visualization
- Training dashboard
- Results analysis
- Image emotion prediction

---

## 🖼️ Image Prediction

From app:
> DL → Image Prediction

Upload an image → Get predicted emotion

---

## 🔧 Tech Stack

- Python 3.12
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy / Pandas
- Matplotlib

---



## ⭐ If you like this project
Give it a star ⭐ on GitHub!
