import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load training and testing datasets
train_data = pd.read_csv('patient_train_data.csv')
test_data = pd.read_csv('patient_test_data.csv')

# Common Preprocessing Function
def preprocess_data(data, fit=True, label_encoder_gender=None, label_encoder_history=None, label_encoder_medication=None):
    if fit:
        label_encoder_gender = {}
        label_encoder_history = {}
        label_encoder_medication = {}

    # Encode gender
    if fit:
        label_encoder_gender = dict(enumerate(data['gender'].astype('category').cat.categories))
    data['gender'] = data['gender'].astype('category').cat.codes

    # Encode medical_history
    if fit:
        label_encoder_history = dict(enumerate(data['medical_history'].astype('category').cat.categories))
    data['medical_history'] = data['medical_history'].astype('category').cat.codes

    # Encode current medications dynamically
    medication_columns = [col for col in data.columns if 'current_medication' in col]
    for col in medication_columns:
        if fit:
            label_encoder_medication[col] = dict(enumerate(data[col].fillna('Unknown').astype('category').cat.categories))
        data[col] = data[col].fillna('Unknown').astype('category').cat.codes

    # Ensure 'age' is treated as a numeric feature
    data['age'] = data['age'].fillna(data['age'].mean())  # Handle missing values by replacing with the mean age
    return data, label_encoder_gender, label_encoder_history, label_encoder_medication

# Preprocess training data
train_data, label_encoder_gender, label_encoder_history, label_encoder_medication = preprocess_data(train_data, fit=True)

# Preprocess testing data (use the same label encoders fitted on training data)
test_data, _, _, _ = preprocess_data(test_data, fit=False, label_encoder_gender=label_encoder_gender, label_encoder_history=label_encoder_history, label_encoder_medication=label_encoder_medication)

# Separate features and labels
X_train = train_data.drop(columns=['patient_id', 'correct_prescription'])
y_train = train_data['correct_prescription']

X_test = test_data.drop(columns=['patient_id', 'correct_prescription'])
y_test = test_data['correct_prescription']

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (with better parameters)
model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Display results
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Bias Analysis
def analyze_bias(data, model):
    demographic_groups = data['gender'].unique()
    fairness_results = {}
    for group in demographic_groups:
        subset = data[data['gender'] == group]
        X_subset = subset.drop(columns=['patient_id', 'correct_prescription'])
        X_subset_scaled = scaler.transform(X_subset)
        y_subset = subset['correct_prescription']
        y_pred_subset = model.predict(X_subset_scaled)
        fairness_results[group] = accuracy_score(y_subset, y_pred_subset)
    return fairness_results

bias_results = analyze_bias(test_data, model)
print("\nBias Analysis Results:", bias_results)

# Misinformation Analysis
def validate_recommendations(predictions, trusted_medical_sources):
    validation_results = [1 if pred in trusted_medical_sources else 0 for pred in predictions]
    return sum(validation_results) / len(validation_results)

trusted_sources = [0, 1]  # Assuming 0 and 1 are both trusted
misinformation_score = validate_recommendations(y_pred, trusted_sources)

print("\nMisinformation Score:", misinformation_score)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()