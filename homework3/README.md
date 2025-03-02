# Homework 3 - Generative Models for Image Processing

This homework explores three major deep learning-based generative models:

1. **Variational Autoencoder (VAE)**: Learning a probabilistic latent space for image reconstruction.
2. **Generative Adversarial Networks (GANs)**: Using adversarial training for image denoising.
3. **Diffusion Models**: Applying diffusion-based techniques for image denoising and generation.

---

## 🚀 Solution Overview

This solution implements and evaluates three different generative models, demonstrating their ability to generate, reconstruct, and denoise images. I utilize PyTorch for the VAE and Diffusion models, while TensorFlow/Keras is used for GANs.

---

## 🔍 Task 1: Variational Autoencoder (VAE)

### 📌 Approach

#### Load and Preprocess Data
- Load the **MNIST** dataset and convert images to grayscale.
- Normalize the data and prepare batches for training.

#### Define the Variational Autoencoder (VAE)
- The **encoder** consists of a series of fully connected layers that compress the image into a **latent space**.
- The **decoder** reconstructs images from the latent space using symmetrical fully connected layers.
- I compute **mean (μ) and log variance (log σ²)** for learning a probabilistic latent space.

#### Reparameterization Trick
- This ensures backpropagation can flow through the stochastic sampling process.

#### Loss Function
- **Reconstruction loss** using **binary cross-entropy**.
- **KL divergence loss** ensures the latent space follows a Gaussian distribution

#### Training and Evaluation
- The model is trained for **10 epochs** with the Adam optimizer.
- **Latent spaces of different sizes (2, 4, 16)** are evaluated.
- **Structural Similarity Index (SSIM) and Mean Squared Error (MSE)** are computed to assess image quality.
- Visualize:
  - Original vs. reconstructed images.
  - 2D **latent space visualization** using **T-SNE** for higher-dimensional latent spaces.

---

## 🎨 Task 2: Generative Adversarial Networks (GANs)

### 📌 Approach

#### Load and Prepare Data
- Use the **Fashion-MNIST** dataset for training.
- Normalize images to **[-1,1]** for stable GAN training.
- Create **noisy images** by adding Gaussian noise.

#### Define the Generator
- Uses **U-Net style** architecture:
  - **Downsampling** layers extract features using **Conv2D + LeakyReLU**.
  - **Upsampling** layers reconstruct the image with **Conv2DTranspose + ReLU**.
  - **Skip connections** help preserve details.
- Outputs a denoised image using a **tanh activation function**.

#### Define the Discriminator
- A **CNN-based classifier** that distinguishes real vs. generated images.
- Uses **binary cross-entropy loss**.

#### Loss Functions
- **Generator Loss**:
  - Adversarial loss (GAN loss).
  - **L1 loss** for structure preservation.
  - **SSIM (Structural Similarity Index)** for perceptual quality.
- **Discriminator Loss**:
  - Measures the ability to distinguish real vs. generated images.

#### Training
- Uses **Adam optimizer** with learning rate **2e-4**.
- Trained for **10 epochs**.
- Periodically saves the model and visualizes generated images.

#### Evaluation
- The generator is used to **denoise images**.
- Side-by-side comparison of **original, noisy, and denoised images**.
- **Test dataset inference** using the trained GAN.

---

## 🌫️ Task 3: Image Denoising with Diffusion Models

### 📌 Approach

#### Load and Preprocess CIFAR-10 Dataset
- Images are loaded and normalized.
- A batch of images is visualized to inspect dataset distribution.

#### Define Diffusion Model
- **Forward Process**:
  - Adds Gaussian noise progressively to the image over **1000 noise steps**.
  - Noise schedule controlled by:
    \[
    x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \epsilon
    \]
- **Reverse Process**:
  - Uses a **U-Net based denoiser** to reconstruct the image.
  - Predicts the added noise and denoises step-by-step.

#### Model Architecture
- **Encoder (Downsampling)**:
  - Uses **Residual Convolution Blocks** with **MaxPooling**.
- **Bottleneck**:
  - A compressed representation of the image.
- **Decoder (Upsampling)**:
  - Uses **Transposed Convolutions** with **skip connections**.
- **Class Conditioning**:
  - Injects class information into the model for controlled generation.

#### Loss Function
- The model is trained to **predict noise** using **Mean Squared Error (MSE)**.

#### Training
- **100 epochs** using **Adam optimizer**.
- Learning rate is decayed over epochs.
- Intermediate model checkpoints are saved.

#### Sampling Process
- Generates new images from random noise.
- Uses **Classifier-Free Guidance** to adjust generation strength.
- Evaluates different noise step settings (**500 vs. 1000 steps**).

#### Evaluation
- Visualizes generated images across **different class labels**.
- Compares performance of **500-step vs. 1000-step models**.

---