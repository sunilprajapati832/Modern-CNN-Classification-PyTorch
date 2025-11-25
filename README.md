# Modern CNN Classification in PyTorch
A deep learning project showcasing modern Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) for image classification using PyTorch. This repository includes training, evaluation, visualization (GradCAM), and comparison of popular architectures. <br>
⚠️ **Note:** This project did not achieve high accuracy due to pipeline issues and incomplete objectives. It is shared here to highlight both the technical work and the lessons learned.

## Dataset - caltech-101 Downlaod link: 
**Caltech-101 Dataset** https://www.kaggle.com/datasets/imbikramsaha/caltech-101
- 9,146 images across 101 object categories + 1 background category.
- Each class contains ~40–800 images.
- Image size ~300 × 200 pixels.
- Categories include objects like airplanes, anchor, butterfly, chair, dolphin, elephant, etc.

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

## Project Structure
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
git clone https://github.com/sunilprajapati832/Modern-CNN-Classification-PyTorch.git    <br>
cd Modern-CNN-Classification-PyTorch                                                    <br>
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
| Model               | Top-1 Accuracy | Top-5 Accuracy | Inference Time (s) | Parameters (M) |
|---------------------|----------------|----------------|---------------------|----------------|
| ResNet50 (pretrained) | 0.55%         | 4.16%          | 137.654             | 23.72          |
| EfficientNet-B0     | 0.82%          | 3.98%          | 67.048              | 4.14           |
| MobileNet-V3        | 0.60%          | 3.13%          | 51.173              | 4.33           |
| ViT-B16             | 4.72%          | 10.77%         | 1140.013            | 85.88          |
| ResNet50 (custom)   | 1.01%          | 4.24%          | 264.243             | 23.72          |

## Requirements
- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

## Why Results Were Poor
- **No data augmentation →** models overfit quickly and failed to generalize.
- **No test set →** only train/validation split, so evaluation wasn’t robust.
- **Short training schedule →** only 10–20 epochs, insufficient for deep models.
- **Low learning rate →** training may have converged too slowly or got stuck.
- **Pipeline instability →** modularity issues caused inconsistencies across models.
- **ViT not tuned →** transformers need careful optimization and large datasets; Caltech‑101 is too small.
- **Pretrained weights mismatch →** custom ResNet required classifier fixes, leading to partial weight loading.

## Lessons Learned
- Importance of clean, modular pipelines for reproducibility
- Need for systematic hyperparameter tuning
- Value of data augmentation (flips, rotations, color jitter)
- Recognizing dataset limitations (Caltech‑101 is small and imbalanced)
- Even failed experiments provide valuable insights into model behavior

## Future Work
- Add support for Optuna hyperparameter tuning
- Extend ROC/PR curves to multi‑class evaluation
- Train with data augmentation and longer schedules
- Deployment with Flask/FastAPI
- Explore larger datasets for ViT benchmarking

**License** This project is licensed under the MIT License (Attribution Required). If you use any part of this project, please give proper credit to the author – Sunil Prajapati.

**If You Like This Project** Please star **⭐** this repository — it helps support my work and increases visibility!

## Connect with Me 🤝
If you found this project interesting, let’s connect!  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow%20Me-blue?logo=linkedin&style=for-the-badge)](https://www.linkedin.com/in/sunil-prajapati832)


