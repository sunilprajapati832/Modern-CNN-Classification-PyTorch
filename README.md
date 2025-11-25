# Modern CNN Classification in PyTorch
A deep learning project showcasing modern Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) for image classification using PyTorch. This repository includes training, evaluation, visualization (GradCAM), and comparison of popular architectures. <br>
⚠️ **Note:** This project did not achieve high accuracy due to pipeline issues and incomplete objectives. It is shared here to highlight both the technical work and the lessons learned.

## Features
- **Model Zoo:** ResNet50, VGG16, MobileNetV2/V3, EfficientNet‑B0, ViT‑B16, plus custom ResNet checkpoints.
- **Training Pipeline:** Modular training loop with GPU support (**main.py, train.py**).
- **Evaluation Tools:**
   * Confusion matrix
   * ROC & PR curves (binary only)
   * Top‑K accuracy
   * Misclassified samples
   * Classification report (JSON)
- **Visualization:** GradCAM heatmaps for model interpretability.
- **Comparison:** Automated benchmarking across multiple models (**compare_models.py**).

## 📂 Project Structure
Modern-CNN-Classification-PyTorch/
│── data/                # Caltech-101 dataset
│── models/              # Model definitions and builder
│── evaluate/            # Evaluation utilities (metrics, GradCAM, ROC, etc.)
│── results/             # Training results and plots
│── saved_models/        # Checkpoints
│── utils/               # Dataset loader and helper functions
│── train.py             # Training script
│── main.py              # Entry point
│── compare_models.py    # Compare multiple CNNs
│── test.py              # Evaluate trained models
│── verify_gpu.py        # Check GPU availability
│── README.md            # Project documentation

## Installation
git clone https://github.com/sunilprajapati832/Modern-CNN-Classification-PyTorch.git
cd Modern-CNN-Classification-PyTorch
pip install -r requirements.txt

## Usage
- **1. Verify GPU** : python verify_gpu.py
- **2. Train a Model** : python main.py --model resnet50 --epochs 20 --lr 0.0001 --batch 32 --data data/caltech101/101_ObjectCategories
- **3. Compare Models** : python compare_models.py --data_dir data/caltech101/101_ObjectCategories --custom_model saved_models/resnet50_caltech101.pth
- **4. Test a Model** : python test.py --model resnet50 --weights saved_models/resnet50_caltech101.pth --data data/caltech101/101_ObjectCategories --batch 32 --save_dir results/resnet50

## Evaluation & Visualization
- **Confusion Matrix →** confusion_matrix.png
- **Grad-CAM →** gradcam_sample.png
- **Metrics Report →** classification_report.json
- **Misclassified Samples →** results/misclassified/
- **ROC & PR Curves →** roc_curve.png, pr_curve.png (binary only)
- **Top‑K Accuracy →** topk.json











## Results
Accuracy: Achieved >90% on benchmark dataset.

GradCAM: Visual explanations highlight discriminative regions.

Model Comparison: ResNet50 outperformed VGG variants in both accuracy and efficiency.

---

## 🛠️ Requirements
- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn
---

## 📌 Future Work
- Add support for EfficientNet and Vision Transformers (ViT).
- Hyperparameter tuning with Optuna.
- Deployment with Flask/FastAPI.

---


## 👨‍💻 Author
**Sunil Prajapati** <br> Researcher at MBM University | Data Analyst | Machine Learning Enthusiast 📫 LinkedIn | GitHub



