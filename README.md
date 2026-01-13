# Dissertation Project – CNN Image Classification

## 1. Project Overview
This project focuses on designing, training, and evaluating a Convolutional Neural Network (CNN) for image classification using a publicly available dataset from Kaggle. The project follows a complete machine learning pipeline, including dataset acquisition, exploratory data analysis, data preprocessing, model development, optimisation, and performance evaluation.

The technical implementation is documented in Jupyter Notebooks, and the repository demonstrates continuous development through regular GitHub commits.

---

## 2. How to Run This Repository
1. Clone the repository from GitHub.
2. Install the required Python packages listed in `requirements.txt`.
3. Download the dataset using the Kaggle API as described in Section 3.
4. Run the notebooks in numerical order inside the `notebooks/` folder.
5. All notebooks are designed to be one-click executable when run in Google Colab or a local Jupyter environment.

---

## 3. Dataset Acquisition (Kaggle)

The dataset used in this project is the **Intel Image Classification** dataset, obtained from Kaggle.

- **Dataset name:** Intel Image Classification  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/puneet6060/intel-image-classification  

The dataset is downloaded programmatically using the Kaggle API within Google Colab. Authentication is performed using a Kaggle API token (`kaggle.json`), which is not stored in the repository for security reasons.

The following commands are used within the dataset acquisition notebook:

```bash
pip install kaggle
kaggle datasets download -d puneet6060/intel-image-classification -p data/ --unzip

## 4. Dataset Description

The Intel Image Classification dataset consists of natural scene images categorised into six distinct classes:

Buildings

Forest

Glacier

Mountain

Sea

Street

The dataset is organised into separate directories for training and testing images. Each class is stored in its own subdirectory, which enables straightforward loading of labelled image data for supervised learning tasks.

The dataset is suitable for convolutional neural networks due to its multi-class structure and visual diversity across categories.

## 5. Project Objective and Success Metrics

[To be completed]

## 6. Exploratory Data Analysis (EDA)

[To be completed]

## 7. Dataset Cleaning

[To be completed]

## 8. Train / Validation / Test Split

[To be completed]

## 9. CNN Model Development
## 9.1 Model 1 – Baseline CNN

[To be completed]

## 9.2 Model 2 – Improved CNN

[To be completed]

## 9.3 Model 3 – Optimised CNN

[To be completed]

## 10. Data Augmentation

[To be completed]

## 11. Model Performance Evaluation

[To be completed]

## 12. Python Packages Used

[To be completed]

## 13. Reused Code and References

[To be completed]

## 14. Conclusions

[To be completed]

## 15. Future Work

[To be completed]

## 16. Known Issues / Bugs

[To be completed]

