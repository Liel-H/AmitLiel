# Project Plan: Iris Machine Learning Classifier

### Phase 1: Data Preparation
1.  **Source Data**: Download the standard Iris dataset (CSV).
2.  **Data Augmentation**: Generate 50 synthetic samples for a 4th species, "lilacina," ensuring feature ranges are distinct yet realistic.
3.  **Integration**: Merge the original and synthetic data into `iris_v2.csv`.

### Phase 2: Model Development
1.  **Scripting**: Write `classifier.py` using Python's `scikit-learn` library.
2.  **Algorithm Selection**: Use `RandomForestClassifier` for robust multi-class performance.
3.  **Data Partitioning**: Implement an 80/20 split (Training/Testing).

### Phase 3: Evaluation & Visualization
1.  **Testing**: Run the model against the hold-out test set.
2.  **Metrics**: Extract accuracy, precision, and recall.
3.  **Visualization**: Generate a 4x4 confusion matrix heatmap using `Seaborn`.

### Phase 4: Documentation
1.  **PRD**: Define project scope and success criteria.
2.  **Report**: Summarize methodology and findings.
3.  **README**: Provide usage instructions and project overview.
