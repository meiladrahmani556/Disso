# Dissertation Project – CNN Image Classification

## 1. Project Overview

This project focuses on the design, training, and evaluation of a Convolutional Neural Network (CNN) for a multi-class image classification task using a publicly available dataset from Kaggle. The project follows a complete machine learning pipeline, including dataset acquisition, exploratory data analysis (EDA), data preprocessing, CNN model development, optimisation, and performance evaluation.

All experiments are implemented in Python using Jupyter Notebooks. The repository demonstrates continuous development through regular GitHub commits and is structured to support reproducibility.

---

## 2. Repository Structure

- `notebooks/` – Jupyter notebooks implementing each stage of the ML pipeline  
- `data/` – Local dataset directory (excluded from version control)  
- `README.md` – Project documentation  

The notebooks are organised numerically to reflect the project workflow.

---

## 3. How to Run This Repository

1. Clone the repository from GitHub.
2. Install the required Python packages listed in `requirements.txt`.
3. Download the dataset using the Kaggle API (see Section 4).
4. Run the notebooks in numerical order inside the `notebooks/` directory.
5. All notebooks are designed to be one-click executable in Google Colab or a local Jupyter environment.

---

## 4. Dataset Acquisition (Kaggle)

The dataset used in this project is the **Intel Image Classification** dataset obtained from Kaggle.

- Dataset name: Intel Image Classification  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification  

The dataset is downloaded programmatically using the Kaggle API within Google Colab. Authentication is performed using a Kaggle API token (`kaggle.json`), which is generated via the Kaggle account settings. For security reasons, this file is not stored in the repository.

The dataset is downloaded and extracted using the following commands:

pip install kaggle  
kaggle datasets download -d puneet6060/intel-image-classification -p data/ --unzip  

The dataset acquisition process is implemented in:

notebooks/01_dataset_acquisition.ipynb

---

## 5. Dataset Description

The Intel Image Classification dataset consists of natural scene images categorised into six classes:

- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street  

The dataset is organised into separate directories for training, testing, and prediction images. Each class is stored in its own subdirectory, enabling supervised learning for multi-class image classification.
---

## 6. Project Objective and Success Metrics

The primary objective of this project is to develop a CNN model capable of accurately classifying natural scene images into one of six predefined categories. Model performance is primarily evaluated using classification accuracy and loss metrics on validation and test datasets.

---

## 7. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was conducted to understand the dataset and inform preprocessing and modelling decisions.

### 7.1 Class Distribution

The number of training images per class was analysed. Results show that all six classes contain a similar number of images, indicating no severe class imbalance. This reduces the need for class weighting during model training.

A bar chart visualising the class distribution is generated in:

notebooks/02_eda.ipynb

### 7.2 Sample Image Inspection

Randomly selected images from each class were visually inspected. The images clearly correspond to their respective class labels, and strong semantic differences between classes such as forest, sea, and street are observable. This confirms the suitability of the dataset for CNN-based image classification.

### 7.3 Image Size Analysis

A random sample of 600 images (100 per class) was analysed to assess image dimensions. All sampled images were found to have consistent dimensions of **150 × 150 pixels**. This uniformity simplifies the preprocessing pipeline, as no additional resizing is required prior to CNN training. Image normalisation is still applied.

EDA is fully implemented in:

notebooks/02_eda.ipynb

---

## 8. Dataset Cleaning and Preprocessing

Images are normalised by rescaling pixel values to the range [0, 1]. The training dataset is split into training (80%) and validation (20%) subsets using Keras ImageDataGenerator. The test dataset is kept separate and is not used during training to ensure unbiased model evaluation.

This stage of the pipeline is implemented in:

notebooks/03_cleaning_and_split.ipynb

---

## 9. CNN Model Development

Three CNN models are developed to demonstrate iterative optimisation:

### 9.1 Model 1 – Baseline CNN
A simple CNN architecture used to establish baseline performance.

### 9.2 Model 2 – Improved CNN
An enhanced CNN model incorporating architectural and training improvements.

### 9.3 Model 3 – Optimised CNN
A final optimised CNN model designed to maximise performance and generalisation.

---

## 10. Data Augmentation
Data augmentation techniques are applied during training to improve robustness and reduce overfitting.

---

## 11. Model Performance Evaluation
Model performance is evaluated using training and validation curves, test set accuracy, and predictions on individual random images.

---

## 12. Python Packages Used
Key Python libraries used in this project include NumPy, Pandas, Matplotlib, TensorFlow/Keras, Scikit-learn, and the Kaggle API.

---

## 13. Reused Code and References
Any reused or adapted code from external tutorials, documentation, or repositories is clearly referenced and acknowledged.

---

## 14. Conclusions
This section summarises key findings and model performance outcomes.

---

## 15. Future Work
Potential improvements include transfer learning, deeper CNN architectures, and deployment as a graphical user interface (GUI) application.

---

## 16. Known Issues / Bugs
Any unresolved issues or limitations encountered during development are documented here.
