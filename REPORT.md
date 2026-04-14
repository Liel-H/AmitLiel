# Iris Classification Project Report (v2)

## 1. Dataset Overview
This project utilizes an expanded version of the classic Iris dataset. The dataset consists of 200 total samples categorized into four species:

- **Iris-setosa**: Smallest petals, easily separable.
- **Iris-versicolor**: Medium size, overlaps slightly with Virginica.
- **Iris-virginica**: Largest petals among the original three.
- **Iris-lilacina (Added)**: Synthetic species with large sepal/petal dimensions.

### Features
Each sample is characterized by 4 features (measured in cm):
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## 2. Model Architecture
The classifier is built using a **Random Forest** algorithm with 100 decision trees. 

### Operational Logic
- **Data Split**: 80% (160 samples) for training, 20% (40 samples) for testing.
- **Training**: The model learns to identify complex decision boundaries between the four species.
- **Testing**: The model predicts species for previously unseen samples to calculate error rates.

---

## 3. Results & Performance
The model demonstrated high reliability on the expanded dataset.

### Performance Summary
- **Overall Accuracy**: **97.50%**
- **Test Samples**: 40
- **Correct Predictions**: 39

### Analysis
The machine correctly identified all Setosa and Versicolor flowers. A single overlap occurred between Virginica and the synthetic Lilacina species, which is consistent with the overlapping feature ranges designed for the experiment.

![Confusion Matrix](confusion_matrix_v2.png)
