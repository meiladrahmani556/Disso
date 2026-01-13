# Dissertation Project – CNN Image Classification

## 1. Project Overview

This project focuses on the design, training, and evaluation of a Convolutional Neural Network (CNN) for an image classification task using a publicly available dataset from Kaggle. The project follows a complete machine learning pipeline, including dataset acquisition, exploratory data analysis (EDA), data preprocessing, CNN model development, optimisation, and performance evaluation.

All experiments are implemented in Python using Jupyter Notebooks. The repository demonstrates continuous development through regular GitHub commits.

---

## 2. How to Run This Repository

1. Clone the repository from GitHub.
2. Install the required Python packages listed in `requirements.txt`.
3. Download the dataset using the Kaggle API (see Section 3).
4. Run the notebooks in numerical order located in the `notebooks/` directory.
5. All notebooks are designed to be one-click executable in Google Colab or a local Jupyter environment.

---

## 3. Dataset Acquisition (Kaggle)

The dataset used in this project is the **Intel Image Classification** dataset obtained from Kaggle.

- Dataset name: Intel Image Classification  
- Source: Kaggle  
- Link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification  

The dataset is downloaded programmatically using the Kaggle API within Google Colab. Authentication is performed using a Kaggle API token (`kaggle.json`), which is generated via the Kaggle account settings. For security reasons, this file is not stored in the repository.

The dataset is downloaded and extracted using the following commands:

pip install kaggle  
kaggle datasets download -d puneet6060/intel-image-classification -p data/ --unzip  

The dataset is stored locally in the `data/` directory, which is excluded from version control using `.gitignore`.

The dataset acquisition process is implemented in:

notebooks/01_dataset_acquisition.ipynb

---

## 4. Dataset Description

The Intel Image Classification dataset consists of natural scene images categorised into six classes:

- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street  

The dataset is organised into separate directories for training, testing, and prediction images. Each class is stored in its own subdirectory, enabling supervised learning for multi-class image classification.

---

## 5. Project Objective and Success Metrics

The primary objective of this project is to develop a CNN model capable of accurately classifying natural scene images into one of six predefined categories. Model performance will be assessed primarily using classification accuracy and loss metrics on validation and test datasets.

---

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was conducted to gain an initial understanding of the dataset and to inform subsequent preprocessing and modelling decisions.

### 6.1 Class Distribution

The number of training images per class was analysed. Results show that all six classes contain a similar number of images, with no severe class imbalance observed. This reduces the need for class weighting during model training.

A bar chart visualising the class distribution is generated in:

notebooks/02_eda.ipynb

### 6.2 Sample Image Inspection

Randomly selected images from each class were visually inspected. The images clearly reflect their respective class labels, and strong semantic differences between classes such as forest, sea, and street are evident. This confirms the suitability of the dataset for a CNN-based image classification task.

### 6.3 Image Size Analysis

A random sample of 600 images (100 per class) was analysed to assess image dimensions. All sampled images were found to have consistent dimensions of **150 × 150 pixels**. This uniformity simplifies the preprocessing pipeline, as no additional resizing is required prior to CNN training. Image normalisation will still be applied.

EDA is fully implemented in:

notebooks/02_eda.ipynb

---

## 7. Dataset Cleaning
Dataset cleaning and preprocessing steps, including normalisation and data preparation for CNN input, are implemented in the subsequent notebooks.

---

## 8. Train / Validation / Test Split
The dataset is split into training, validation, and test sets to enable robust model evaluation and hyperparameter tuning.

---

## 9. CNN Model Development

Three CNN models are developed to demonstrate iterative optimisation:

### 9.1 Model 1 – Baseline CNN
A simple CNN architecture used to establish baseline performance.

### 9.2 Model 2 – Improved CNN
An enhanced model incorporating architectural and training improvements.

### 9.3 Model 3 – Optimised CNN
A final optimised CNN model designed to improve generalisation and performance.

---

## 10. Data Augmentation
Data augmentation techniques are applied during training to improve model robustness and reduce overfitting.

---

## 11. Model Performance Evaluation
Model performance is evaluated using training and validation curves, test set accuracy, and predictions on individual random input images.

---

## 12. Python Packages Used
Key Python libraries used in this project include NumPy, Pandas, Matplotlib, TensorFlow/Keras, Scikit-learn, and the Kaggle API.

---

## 13. Reused Code and References
Any reused or adapted code from external sources is appropriately referenced and acknowledged.

---

## 14. Conclusions
This section summarises key findings and performance outcomes.

---

## 15. Future Work
Potential future improvements include transfer learning, deeper architectures, and deployment as a graphical user interface (GUI) application.

---

## 16. Known Issues / Bugs
Any unresolved issues or limitations encountered during development are documented here.
