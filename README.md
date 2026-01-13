# Dissertation Project – CNN Image Classification

## 1. Project Overview

This project focuses on the design, training, and evaluation of a Convolutional Neural Network (CNN) for an image classification task using a publicly available dataset from Kaggle. The project follows a complete machine learning pipeline, beginning with dataset acquisition and exploratory data analysis, and progressing towards model development, optimisation, and performance evaluation.

The technical implementation is carried out using Python in Jupyter Notebooks, and the repository demonstrates continuous development through regular GitHub commits.

---

## 2. How to Run This Repository

1. Clone this GitHub repository.
2. Install the required Python packages listed in `requirements.txt`.
3. Download the dataset using the Kaggle API as described in Section 3.
4. Run the notebooks in numerical order located in the `notebooks/` directory.
5. All notebooks are designed to be one-click executable when run in Google Colab or a local Jupyter environment.

---

## 3. Dataset Acquisition (Kaggle)

The dataset used in this project is the Intel Image Classification dataset, obtained from Kaggle.

- Dataset name: Intel Image Classification  
- Source: Kaggle  
- Dataset link: https://www.kaggle.com/datasets/puneet6060/intel-image-classification  

The dataset is downloaded programmatically using the Kaggle API within Google Colab. Authentication is performed using a Kaggle API token (kaggle.json), which is generated through the Kaggle account settings. For security reasons, this file is not stored in the repository.

The following commands are used within the dataset acquisition notebook to download and extract the dataset:

pip install kaggle  
kaggle datasets download -d puneet6060/intel-image-classification -p data/ --unzip

The dataset is stored locally in the data directory, which is excluded from version control using .gitignore.

The dataset acquisition process is implemented in:

notebooks/01_dataset_acquisition.ipynb

---

## 4. Dataset Description

The Intel Image Classification dataset consists of natural scene images grouped into six distinct classes:

- Buildings  
- Forest  
- Glacier  
- Mountain  
- Sea  
- Street  

The dataset is organised into separate directories for training and testing images. Each class is stored in its own subdirectory, allowing straightforward loading of labelled image data for supervised learning tasks.

The dataset contains visually diverse images with varying lighting conditions, backgrounds, and resolutions, making it suitable for evaluating convolutional neural networks on real-world image classification problems.

---

## 5. Project Objective and Success Metrics

The primary objective of this project is to develop a CNN model capable of accurately classifying natural scene images into one of the six predefined classes.

Model performance will be evaluated using classification accuracy and loss metrics on validation and test datasets. Additional evaluation methods, such as visual inspection of predictions on individual images, will be used to assess model behaviour.

---

## 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is conducted to gain an initial understanding of the dataset. This includes:

- Identifying the available classes  
- Examining the distribution of images across classes  
- Visual inspection of sample images  
- Checking image resolutions and variability  

EDA is implemented in:

notebooks/02_eda.ipynb

---

## 7. Dataset Cleaning

Dataset cleaning and preprocessing steps are performed to prepare the data for model training. This includes resizing images, normalisation, and handling any corrupted or inconsistent files.

---

## 8. Train / Validation / Test Split

The dataset is split into training, validation, and test sets to enable model training, hyperparameter tuning, and unbiased performance evaluation. The rationale behind the chosen split ratios is documented within the relevant notebook.

---

## 9. CNN Model Development

Multiple CNN models are developed and evaluated to demonstrate iterative improvement and optimisation.

### 9.1 Model 1 – Baseline CNN

A simple baseline CNN architecture is implemented to establish initial performance.

### 9.2 Model 2 – Improved CNN

The second model incorporates architectural or hyperparameter improvements based on observations from the baseline model.

### 9.3 Model 3 – Optimised CNN

The final model applies further optimisation techniques to improve generalisation and overall performance.

---

## 10. Data Augmentation

Data augmentation techniques are applied during training to improve model robustness and reduce overfitting. The selected augmentation methods and their impact on performance are discussed.

---

## 11. Model Performance Evaluation

Model performance is assessed using training and validation curves, test set metrics, and predictions on randomly selected individual images. These results are visualised and analysed.

---

## 12. Python Packages Used

Key Python packages used in this project include:

- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- TensorFlow / Keras  
- Kaggle API  

A full list of dependencies is provided in requirements.txt.

---

## 13. Reused Code and References

Any reused or adapted code from external tutorials, documentation, or repositories is clearly referenced and acknowledged in this section.

---

## 14. Conclusions

This section summarises the key findings of the project, including model performance and insights gained from experimentation.

---

## 15. Future Work

Potential improvements and extensions to the project include exploring transfer learning, training on larger datasets, and deploying the trained model as a graphical application.

---

## 16. Known Issues / Bugs

Any unresolved issues or limitations encountered during development are documented here.
