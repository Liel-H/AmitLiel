import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the original dataset
data = pd.read_csv('iris.csv')

# Invent data for a 4th species: "lilacina"
np.random.seed(42)
new_species_data = pd.DataFrame({
    'sepal_length': np.random.normal(7.5, 0.4, 50),
    'sepal_width': np.random.normal(3.0, 0.3, 50),
    'petal_length': np.random.normal(6.5, 0.5, 50),
    'petal_width': np.random.normal(2.3, 0.2, 50),
    'species': ['lilacina'] * 50
})

# Append new data to original
data_v2 = pd.concat([data, new_species_data], ignore_index=True)

# Save the expanded dataset
data_v2.to_csv('iris_v2.csv', index=False)
print("Updated dataset saved as 'iris_v2.csv'")

# Split features (X) and target (y)
X = data_v2.drop('species', axis=1)
y = data_v2['species']

# Split 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Build and train the classifier (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the machine
y_pred = model.predict(X_test)

# Report results
print(f"Accuracy with 4 species: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix (4x4)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.title('Confusion Matrix (4x4) - Iris Dataset v2')
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')

# Save as PNG
plt.savefig('confusion_matrix_v2.png')
print("\nNew confusion matrix (4x4) saved as 'confusion_matrix_v2.png'")
