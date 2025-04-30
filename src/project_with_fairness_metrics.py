import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import MetricFrame, selection_rate as fair_accuracy, selection_rate
from sklearn.linear_model import LogisticRegression

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

    # Handle missing age values
    data['age'] = data['age'].fillna(data['age'].mean())

    return data, label_encoder_gender, label_encoder_history, label_encoder_medication

# Preprocess datasets
train_data, label_encoder_gender, label_encoder_history, label_encoder_medication = preprocess_data(train_data, fit=True)
test_data, _, _, _ = preprocess_data(test_data, fit=False, label_encoder_gender=label_encoder_gender, label_encoder_history=label_encoder_history, label_encoder_medication=label_encoder_medication)

# Features and labels
X_train = train_data.drop(columns=['patient_id', 'correct_prescription'])
y_train = train_data['correct_prescription']

X_test = test_data.drop(columns=['patient_id', 'correct_prescription'])
y_test = test_data['correct_prescription']

# Extract gender for fairness constraint
A_train = train_data['gender']
A_test = test_data['gender']

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Base model: Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
gbc.fit(X_train_scaled, y_train)

# Fairness-aware model using post-processing
mitigator = ThresholdOptimizer(
    estimator=gbc,
    constraints="equalized_odds",
    prefit=True,
    predict_method="auto"
)
mitigator.fit(X_train_scaled, y_train, sensitive_features=A_train)

# Fair predictions
y_pred = mitigator.predict(X_test_scaled, sensitive_features=A_test)

# Evaluation
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Fairness Metrics
metric_frame = MetricFrame(
    metrics={"accuracy": fair_accuracy, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test
)

print("\n=== Fairness Metrics by Gender ===")
print(metric_frame.by_group)

# Misinformation Analysis
def validate_recommendations(predictions, trusted_sources):
    return sum(1 for pred in predictions if pred in trusted_sources) / len(predictions)

trusted_sources = [0, 1]  # Example trusted prescription codes
misinformation_score = validate_recommendations(y_pred, trusted_sources)
print(f"\nMisinformation Score: {misinformation_score:.2f}")
