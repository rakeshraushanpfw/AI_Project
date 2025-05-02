**Fair AI-Based Prescription System**
This project implements a machine learning pipeline to assist in generating accurate and fair medical prescriptions. 
Using a Gradient Boosting Classifier (GBC) alongside the Fairlearn framework, the model mitigates gender bias 
while maintaining high accuracy. The system also integrates bias evaluation, misinformation filtering, 
and fairness-aware threshold optimization.

**Key Features**
Gradient Boosting Classifier (GBC): Used for high-accuracy prescription predictions.

Bias Mitigation: Uses Fairlearn's ThresholdOptimizer for equalized odds fairness across gender groups.

Misinformation Filter: Validates predictions against trusted medication sources.

**Performance Evaluation:**

Accuracy, precision, recall, F1 score

Confusion matrix

Fairness metrics by gender (accuracy and selection rate)

**Results**
Accuracy: 99%

Gender Bias: Post-fairness correction reduced performance gap between male and female groups.

Misinformation Score: 100% alignment with trusted sources.

**Future Work**
Expand to larger and more diverse datasets.

Integrate richer clinical and contextual features.

Dynamically connect to updated medical databases and clinical guidelines.

**Visualizations**
Confusion Matrix

Gender Fairness Metrics Graph

Performance Metric Comparisons

**Dependencies**
Python 3.8+

pandas

numpy

scikit-learn

fairlearn

matplotlib

seaborn

**GitHub Repository**
https://github.com/rakeshraushanpfw/AI_Project

