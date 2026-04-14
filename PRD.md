# Product Requirements Document (PRD): Iris Classification Project

## 1. Project Overview
The goal of this project is to develop an automated classification machine capable of identifying four distinct species of Iris flowers based on physical measurements. This project is part of the "AI & Machine Learning" course (Assignment 1).

## 2. Objectives
- Implement a machine learning classifier to distinguish between four Iris species.
- Utilize a dataset with four features (Sepal Length, Sepal Width, Petal Length, Petal Width).
- Achieve high accuracy using an 80/20 training-to-testing data split.
- Document the entire process in a comprehensive summary report.

## 3. Functional Requirements
### 3.1 Data Management
- **Dataset Source:** Use the standard Iris dataset as a base.
- **Data Augmentation:** Invent a fourth Iris species with reasonable feature ranges to expand the dataset to four categories.
- **Data Splitting:** Implement a mechanism to split the dataset into 80% for training and 20% for testing.

### 3.2 Classification Machine
- **Algorithm:** Use a suitable classification algorithm (e.g., Random Forest, k-NN).
- **Implementation:** The machine must be built using CLI-based tools (Gemini CLI / Claude CLI).
- **Validation:** The machine must be tested against the 20% hold-out set to verify performance.

### 3.3 Visualizations & Output
- **Confusion Matrix:** Generate a 4x4 confusion matrix visualizing the prediction results.
- **Performance Metrics:** Calculate accuracy, precision, recall, and f1-score.
- **Reports:** Generate a summary report (Markdown or PDF) including:
    - Dataset description (categories, features, structure).
    - Machine architecture and operational logic.
    - Confusion Matrix visualization.
    - Accuracy and performance analysis.

## 4. Technical Requirements
- **Language:** Python or JavaScript (Node.js).
- **Libraries:** Pandas, Scikit-learn, Matplotlib/Seaborn (for Python) or equivalent for Node.js.
- **Version Control:** All code and documentation must be stored in a public GitHub repository.

## 5. Success Criteria
- Successful training of the model on 4 species.
- Generation of a 4x4 confusion matrix.
- Accuracy exceeding 90% on the test set.
- Completed PRD, Code, and Final Report stored in the project folder.

---
*Created by Gemini CLI - Based on Dr. Yoram Segal's ML Summary*
