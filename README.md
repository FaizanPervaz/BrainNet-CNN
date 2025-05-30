# BrainNet-CNN
🚀 Deep Learning-based image classification project using Convolutional Neural Networks (CNNs) for medical image analysis. Trained on a multi-class dataset with advanced metrics like AUC, F1 Score, and precision-recall evaluation. Includes Grad-CAM visualization and performance tuning with Adam optimizer.

# 🧠 BrainNet-CNN
A Convolutional Neural Network (CNN)-based deep learning model designed for accurate and interpretable **multi-class image classification**. This project focuses on classifying brain scan images into three categories using TensorFlow/Keras and incorporates modern best practices including data augmentation, performance metrics (AUC, F1, precision, recall), and Grad-CAM visualizations for model explainability.

## 📂 Dataset

- The dataset is organized into **three folders (classes)** inside a root directory:
- All images are resized to `150x150` pixels and normalized before being passed to the model.
- Healthy Images were taken from this dataset
  **https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection**
- HGG and LGG were taken as a mix from these datasets 
  **https://brain-development.org/ixi-dataset/**
  **https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019**  
  
## 🚀 Features

- 📦 **Data Loading using ImageDataGenerator**  
- 🧠 **7-layer CNN architecture with ReLU activations**
- 🧪 **Evaluation Metrics:** Accuracy, AUC, Precision, Recall, F1 Score  
- 🏋️‍♂️ **Optimized with Adam Optimizer**
- 🧯 **Dropout & ReduceLROnPlateau for overfitting control**
- 🔥 **Grad-CAM heatmaps for interpretability**

## 📊 Results

- Achieved **training accuracy ~99.7%**, **AUC ~0.999**
- Tuned the model to generalize well by reducing overfitting (target validation accuracy ~97%)

## 🧪 Model Architecture
  Conv2D → ReLU → MaxPool2D
        ↓
  Conv2D → ReLU → MaxPool2D
        ↓
  ... (Total 5 Conv Blocks)
        ↓
  Dropout(0.1)
        ↓
  Dense(512) → ReLU
        ↓
  Dense(3) → Softmax
  
## 🧠 Grad-CAM Visualization
Grad-CAM heatmaps are used to visualize which regions of the image influenced the model's decision. Helpful in medical imaging to identify critical features.

## 🧮 Evaluation Metrics Used
-✅ **Accuracy**
-🧪 **AUC**
-📐 **Precision**
-📉 **Recall**  
-🧬 **Macro F1 Score**

## 📁 Folder Structure
BrainNet-CNN/
├── dataset/
│   ├── Healthy/
│   ├── HGG/
│   └── LGG/
├── models/
├── visualizations/
├── src/
│   ├── train.py
│   ├── grad_cam.py
│   └── utils.py
├── README.md
└── requirements.txt

## 📌 Requirements
- Python 3.10.16
- TensorFlow ≥ 2.8
- Keras
- NumPy
- Matplotlib
- scikit-learn
- tensorflow-addons
- PIL
