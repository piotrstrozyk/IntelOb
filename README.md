# Computational Intelligence Projects

This repository contains my projects related to computational intelligence, including image recognition, clustering optimization, and natural language processing (NLP) exercises.

Project folders contain jupyter notebooks (.ipynb files) which provide detailed explanations in a readable format.

### 📝 Labs
The Labs folder contains various exercises conducted during this course, utilizing technologies such as:  
- **Genetic Algorithms (GA)**  
- **Swarm Intelligence** (PSO, ACO)  
- **Text Generation**  
- **Social Media Scraping**  
- **Object Detection**  
- **Transfer Learning**  
- **Convolutional Neural Networks (CNNs)**  


## 📂 Projects

### 🛩️ Image Recognition Project
This project focuses on image classification using a dataset of military aircraft images from Kaggle. The goal is to train models to accurately classify different types of aircraft.

#### 🔹 Dataset
- Contains **49 different aircraft types** with a total of **23,654 images**.
- Images are resized to **128x128 pixels** for efficient processing.

#### 🔹 Technologies & Methods
- **Transfer Learning Models Used**:
  - ResNet50
  - EfficientNetB3 (Best Performance)
  - Xception
- **Neural Network Techniques**:
  - `GlobalAveragePooling2D`, `Dense`, `Dropout`, `BatchNormalization`
  - Data augmentation: `RandomRotation`, `RandomFlip`
- **Optimization**:
  - Adamax optimizer for ResNet50 & EfficientNetB3
  - RMSprop optimizer for Xception

#### 🔹 Results
- **EfficientNetB3** achieved the highest accuracy with minimal overfitting.
- **ResNet50** showed early overfitting.
- **Xception** had the lowest accuracy.

#### 🔹 Generative Adversarial Network (GAN)
- Attempted to generate aircraft images using GANs.
- Preprocessing included grayscale conversion, resizing, and normalization.
- Generator & discriminator models were implemented, but results did not produce realistic aircraft images.

#### 🔹 Conclusion
This project successfully implemented transfer learning for image classification, with **EfficientNetB3** being the best-performing model. GAN-based image generation remains an area for improvement.

---

### 🔬 Cluster Stability Project
This project explores the stability of **particle clusters** based on the **Morse potential**, which models atomic interactions.

#### 🔹 Morse Potential
- Defines interaction energy between two particles based on distance.
- Total energy is the sum of all particle-pair interactions.

#### 🔹 Optimization Algorithms
- **Genetic Algorithm (GA)**:
  - Uses selection, crossover, mutation, and local optimization.
- **Particle Swarm Optimization (PSO)**:
  - Swarm-based approach updating positions based on best configurations.
  - Includes local optimization for refinement.

#### 🔹 Technologies & Libraries
- **Python**, **NumPy**: Core computations.
- **Matplotlib**, **Plotly**: Data visualization.
- **SciPy**: Optimization functions.
- **py3Dmol**: 3D visualization of particle clusters.

#### 🔹 Experiments
- Conducted for different **ρ values** (3, 6, 10, 14) and **particle counts (N = 5, 6, 7, 8)**.
- Results compare **GA** vs. **PSO** performance.

#### 🔹 Visualization
- 3D representations of optimized particle clusters using **py3Dmol** and **Plotly**.

#### 🔹 Conclusion
This project demonstrates **GA and PSO**'s effectiveness in optimizing Morse potential energy for stable particle clusters. Both algorithms benefit from local optimization enhancements.

---
