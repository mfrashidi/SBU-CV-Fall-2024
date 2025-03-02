# Homework 2 - Deep Learning for Image Processing

This homework explores three fundamental deep learning-based computer vision tasks:

1. **Shoe Brand Classification Using CNN and ResNet**: Classifying shoe images using deep convolutional networks.
2. **Image Denoising Using U-Net**: Removing noise from images using a U-Net architecture.
3. **Airplane Detection Using Selective Search and Faster R-CNN**: Detecting airplanes in images using classical and deep learning-based object detection methods.

---

## 🚀 Solution Overview

This solution leverages deep learning techniques, including Convolutional Neural Networks (CNNs), ResNet, U-Net, and Faster R-CNN, to solve classification, image restoration, and object detection tasks.

---

## 👟 Task 1: Shoe Brand Classification Using CNN and ResNet

### 📌 Overview
This task classifies shoe brands (**Nike, Adidas, Converse**) using:
1. **Custom CNN** – A convolutional model trained from scratch.
2. **Enhanced ResNet-50** – A fine-tuned pretrained model with transfer learning.

#### Data Preparation
- **Augmentation:** Resized, horizontally flipped, and normalized images.
- **Data Loaders:** Train/Test datasets loaded using `ImageFolder` with mini-batches.

#### Model 1: Custom CNN
- **Architecture:** Stacked **convolutional layers**, batch normalization, dropout, and fully connected layers.
- **Training:** 
  - **Optimizer:** SGD (`lr=0.001`)
  - **Loss Function:** Cross-Entropy Loss
  - **Epochs:** 20
- **Tuning:** Experimented with **learning rate** (`0.001` vs `0.01`) and **dropout rates**.

#### Model 2: Enhanced ResNet-50
- **Modifications:** Fine-tuned **pretrained ResNet-50** by replacing the FC layer.
- **Augmentation:** Duplicated dataset with **flipped images**.
- **Training:** 
  - **Optimizer:** Adam (`lr=0.0001`)
  - **Epochs:** 30
  - **Dropout:** 0.3 to prevent overfitting.
 
#### Results & Comparison

| Model | Learning Rate | Dropout | Test Accuracy |
|-------|-------------|---------|--------------|
| **CNN** | 0.001 | 0.1 | **54.39%** |
| **CNN (Higher LR)** | 0.01 | 0.1 | **52.63%** |
| **CNN (Higher Dropout)** | 0.001 | 0.3 | **62.28%** |
| **ResNet-50** | 0.0001 | 0.3 | **88.09%** |

- **ResNet-50** outperformed all CNN models.
- **Data augmentation** improved accuracy.
- **Transfer learning** was key to achieving **high accuracy (>86%)**.

#### Visuals
- **Training Loss Trends** – CNN vs. ResNet stability.
- **Misclassification Analysis** – Confusion matrix to identify model weaknesses.

---

## 🎭 Task 2: Image Denoising Using U-Net

### 📌 Overview
This task involves removing **Salt-and-Pepper noise** from images using a **U-Net** model. The performance is evaluated using **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index).**

#### Data Preparation
- **Noise Addition**: Applied **Salt-and-Pepper noise** (0.07 probability).
- **Preprocessing**:
  - Resized images to `64x64`.
  - Normalized to range `[-1,1]`.
- **Data Loaders**:
  - Split train data into **80% training / 20% validation**.
  - Test dataset loaded separately.

#### U-Net Architecture
- **Encoder**:
  - **5 convolutional blocks** with increasing feature maps (`64 → 128 → 256 → 512 → 1024`).
- **Decoder**:
  - **4 deconvolutional blocks** with skip connections.
- **Final Output**:
  - 3-channel output image (same as input).

#### Training Process
- **Loss Function**: Mean Squared Error (MSE Loss).
- **Optimizer**: Adam (`lr=0.001`).
- **Epochs**: 20.
- **Training Strategy**:
  - **Validation loss tracking** for best model checkpointing.
  - **Early stopping** based on lowest validation loss.

#### Results & Evaluation
- **Metrics**:
  - **PSNR**: Measures signal strength relative to noise.
  - **SSIM**: Evaluates structural similarity to the original image.

- **U-Net significantly improves image quality**, restoring details lost due to noise.

#### Visual Comparisons
- **Original vs. Noisy vs. U-Net Output** displayed side by side.
- **PSNR & SSIM trends** plotted for 10 sample images.

---

## ✈️ Task 3: Airplane Detection Using Selective Search and Faster R-CNN

### 📌 Overview
This task involves detecting airplanes in images using two approaches:
1. **Selective Search + VGG16 Classifier** – A region proposal-based method.
2. **Faster R-CNN** – A deep learning-based object detection model.

#### Data Preparation
- **Dataset**: Airplane images with bounding box annotations.
- **Preprocessing**:
  - Images resized to `224x224`.
  - Normalized pixel values.
- **Data Augmentation**:
  - **Flipping & Rotation** to improve model generalization.

#### Approach 1: Selective Search + VGG16
- **Region Proposal**:
  - **Selective Search** generates bounding box proposals.
  - **IoU Calculation** filters boxes based on overlap with ground truth:
    - **IoU > 0.7** → **Positive samples (airplane)**.
    - **IoU < 0.3** → **Negative samples (background)**.
- **VGG16 Classifier**:
  - Pretrained **VGG16** model with fine-tuning.
  - Final classification layer predicts **airplane vs. background**.
- **Training**:
  - **Optimizer:** Adam (`lr=0.0001`).
  - **Loss Function:** Categorical Cross-Entropy.
  - **Early Stopping & Checkpointing** to prevent overfitting.

#### Approach 2: Faster R-CNN (Deep Learning-Based)
- **Dataset Preparation**:
  - Custom dataset class created using `torchvision.transforms`.
  - Images paired with bounding box annotations.
- **Model Architecture**:
  - **Pretrained Faster R-CNN (ResNet-50 FPN)**.
  - Modified **classification head** to detect airplanes.
- **Training**:
  - **Optimizer:** SGD (`lr=0.001`, `momentum=0.9`).
  - **Learning Rate Scheduler** for gradual adjustments.
  - **Epochs:** 20.

#### Results & Evaluation
- **Metrics**:
  - **Precision**: Measures detection accuracy.
  - **Recall**: Measures how well the model finds all airplanes.
  - **F1-Score**: Balances precision and recall.
  - **IoU (Intersection over Union)**: Evaluates bounding box overlap.

- **Faster R-CNN outperforms Selective Search**, achieving higher accuracy and better localization.

#### Visualizations
- **Selective Search Region Proposals** – Displayed on sample images.
- **Loss & Accuracy Trends** – Plotted for both models.
- **Detection Results** – Ground truth vs. model-predicted bounding boxes.
