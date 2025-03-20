import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Simulated dataset
np.random.seed(42)
data = pd.DataFrame({
    'patient_id': range(1, 11),
    'age': np.random.randint(20, 80, 10),
    'gender': np.random.choice(['Male', 'Female'], 10),
    'prescribed_drug': np.random.choice(['DrugA', 'DrugB', 'DrugC', 'DrugD'], 10),
    'known_allergies': np.random.choice([0, 1], 10),
    'drug_interactions': np.random.choice([0, 1], 10),
    'correct_prescription': np.random.choice([0, 1], 10)
})

# Encoding categorical variables
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])  # Converts Male/Female to 0/1

# Preprocessing
X = data.drop(columns=['correct_prescription', 'patient_id', 'prescribed_drug'])  # Remove non-numeric column
y = data['correct_prescription']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Store results in DataFrame
current_results = pd.DataFrame({
    'Metric': ['Model Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, classification_rep['weighted avg']['precision'], classification_rep['weighted avg']['recall'], classification_rep['weighted avg']['f1-score']]
})

# Fairness & Bias Analysis
def analyze_bias(data, model):
    demographic_groups = data['gender'].unique()
    fairness_results = {}
    for group in demographic_groups:
        subset = data[data['gender'] == group]
        X_subset = subset.drop(columns=['correct_prescription', 'patient_id', 'prescribed_drug'])
        y_subset = subset['correct_prescription']
        X_subset_scaled = scaler.transform(X_subset)
        y_pred_subset = model.predict(X_subset_scaled)
        fairness_results[label_encoder.inverse_transform([group])[0]] = accuracy_score(y_subset, y_pred_subset)
    return fairness_results

bias_results = analyze_bias(data, model)
bias_results_df = pd.DataFrame.from_dict(bias_results, orient='index', columns=['Accuracy'])

# Misinformation Analysis
def validate_recommendations(predictions, trusted_medical_sources):
    validation_results = [1 if pred in trusted_medical_sources else 0 for pred in predictions]
    return sum(validation_results) / len(validation_results)

trusted_sources = [0, 1]  # Placeholder for valid prescriptions (1 = correct, 0 = incorrect)
misinformation_score = validate_recommendations(y_pred, trusted_sources)

misinformation_results = pd.DataFrame({
    'Metric': ['Misinformation Score'],
    'Value': [misinformation_score]
})

# Display test results
test_results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

# Current Results Summary
detailed_current_results = pd.DataFrame({
    'Aspect': ['Model Training & Testing', 'Bias Analysis', 'Misinformation Analysis'],
    'Details': [
        f'Accuracy: {accuracy:.2f}',
        f'Accuracy across demographics: {bias_results_df.to_dict()}',
        f'Misinformation score: {misinformation_score:.2f}'
    ]
})

# Upcoming Results
upcoming_results = pd.DataFrame({
    'Next Steps': [
        'Fine-tune hyperparameters to improve accuracy',
        'Implement re-weighting techniques to mitigate bias',
        'Enhance misinformation detection mechanism'
    ]
})

# Print Results
print("Current Results:")
print(detailed_current_results.to_string(index=False))

print("\nUpcoming Results:")
print(upcoming_results.to_string(index=False))

print("\nBias Analysis Results:")
print(bias_results_df.to_string())

print("\nMisinformation Results:")
print(misinformation_results.to_string(index=False))

print("\nTest Results:")
print(test_results.to_string(index=False))
