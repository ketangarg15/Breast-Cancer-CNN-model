# Breast Cancer Classification with ResNet50

This project demonstrates the use of **ResNet50**, a pre-trained deep learning model, for classifying breast cancer histology images into two categories: **Benign** and **Malignant**. It includes preprocessing the dataset, training a frozen ResNet50 model, and fine-tuning the model for improved performance.

---

## Dataset
The dataset used is the **BreaKHis** dataset of breast cancer histology images. The dataset contains two classes:
- **Benign**
- **Malignant**

The dataset is split into training (80%) and validation (20%) sets for model evaluation.

---

## Features
- **Dataset Preprocessing**: Organizes images into `train` and `validation` directories with a stratified split.
- **Data Augmentation**: Uses rescaling and augmentation for robust training.
- **Transfer Learning**: Employs ResNet50 (pre-trained on ImageNet) as a base model.
- **Fine-Tuning**: Fine-tunes the ResNet50 layers to enhance performance.
- **Evaluation**: Includes metrics such as accuracy, classification report, and AUC-ROC score.
- **Visualization**: Plots training accuracy and loss curves.

---

## Dependencies
- Python >= 3.7
- TensorFlow >= 2.4.0
- scikit-learn
- matplotlib
- Kaggle dataset library (if applicable)

Install the required packages using:
```bash
pip install tensorflow scikit-learn matplotlib
