# Homework 1 - 📷 Computer Vision Tasks

This homework consists of three key computer vision challenges:

1. **License Plate Detection**: Detecting and localizing license plates in images.
2. **Star Detection and Localization**: Identifying and mapping stars in astronomical images.
3. **Image Noise Simulation and Denoising Evaluation**: Simulating noise in images and evaluating different denoising techniques.

---

## 🚀 Solution Overview

This solution employs a combination of classical computer vision techniques and modern machine learning-based approaches to address the given tasks.

---

## 🔍 Task 1: License Plate Detection

### 📌 Approach

#### Load and Preprocess Image
- Load the input image and convert it to grayscale.
- Crop the region of interest to focus on the license plate.

#### Edge Detection
- Apply Canny edge detection to highlight prominent edges.

#### Extract License Plate Contour
- Identify and sort contours based on area.
- Extract the largest quadrilateral contour that likely represents the license plate.

#### Apply Mask and Isolate License Plate
- Create a mask to isolate the detected license plate region.
- Apply bitwise operations to extract the plate.

#### Extract Text from License Plate
- Crop the detected plate region.
- Use Optical Character Recognition (OCR) to extract the license plate number.

---

## ✨ Task 2: Star Detection and Localization

### Approach

#### Load Image
- Read image data from a **text file** containing pixel values.
- Convert pixel data into a structured **NumPy array**.

#### Convert to Grayscale and Thresholding
- Convert the image to grayscale.
- Apply **thresholding** to extract bright regions corresponding to stars.

#### Detect Stars
- Identify **contours** in the thresholded image.
- Compute **centroids** of detected stars.

#### Highlight and Count Stars
- Draw contours around detected stars.
- Store and display **star coordinates** and **total count**.

---

## 🎭 Task 3: Image Noise Simulation and Denoising Evaluation

### Approach

#### Load Images
- Load multiple test images for noise simulation and denoising experiments.

#### Evaluate Image Denoising (SSIM & PSNR)
- Add **Gaussian noise** to images with varying standard deviations.
- Apply **Non-Local Means Denoising** to remove noise.
- Compute **Structural Similarity Index (SSIM)** and **Peak Signal-to-Noise Ratio (PSNR)** for evaluating the effectiveness of denoising.

#### Visualizing SSIM and PSNR Trends
- Plot **SSIM** and **PSNR** against increasing noise levels to analyze the degradation and recovery of image quality.

#### Effect of Gaussian Noise and Denoising on Images
- Apply **Gaussian noise** to images.
- Display **original, noisy, and denoised images** side by side for comparison.

---

## ⚡ Additional Task: Comparing CPU vs GPU Processing

### Approach

#### Check PyTorch & GPU Availability
- Verify the PyTorch version.
- Check if **Metal Performance Shaders (MPS)** (Apple's GPU acceleration) is available.
- Set the processing device (`CPU` or `GPU`).

#### Load and Preprocess Image
- Load the image and convert it to grayscale.
- Simulate **Gaussian noise** on the image.

#### Apply Non-Local Means Denoising
- Use **nlm2d (Non-Local Means Denoising)** on both **CPU** and **GPU**.
- Measure and compare the **processing time** for both.

#### Visualizing Performance Comparison
- Display denoised images processed on **CPU** and **GPU**.
- Annotate the images with the respective **processing times**.

---