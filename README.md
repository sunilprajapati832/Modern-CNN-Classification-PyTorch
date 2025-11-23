# Modern CNN Classification in PyTorch 🧠📊

A deep learning project showcasing **modern Convolutional Neural Networks (CNNs)** for image classification using **PyTorch**.  
This repository includes training, evaluation, visualization (GradCAM), and comparison of popular CNN architectures.

---

## 🚀 Features
- Model Zoo: Implementations of AlexNet, VGG16, VGG19, ResNet50, and custom CNNs.
- Training Pipeline: Modular training loop with GPU support.
- Evaluation Tools:
  - Confusion matrix
  - ROC & PR curves
  - Top‑K accuracy
  - Misclassified samples
- Visualization: GradCAM heatmaps for model interpretability.
- Comparison: Automated scripts to compare multiple models on the same dataset.

---

## 📂 Project Structure
Modern-CNN-Classification-PyTorch/ │── data/ # Dataset folder (ignored in .gitignore) │── models/ # Model definitions and builder │── evaluate/ # Evaluation utilities (metrics, GradCAM, ROC, etc.) │── results/ # Training results and plots │── saved_models/ # Checkpoints │── utils/ # Dataset loader and helper functions │── train.py # Training script │── main.py # Entry point │── compare_models.py # Compare multiple CNNs │── run_gradcam_only.py # Run GradCAM visualization │── verify_gpu.py # Check GPU availability │── README.md # Project documentation


---


## 📊 Results
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
