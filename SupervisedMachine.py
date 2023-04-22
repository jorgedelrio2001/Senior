import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('heart_cleveland_upload.csv')

# Preprocess data
X = data.drop('condition', axis=1)
y = data['condition']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=5000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Print evaluation metrics
print("Logistic Regression Accuracy: {:.2f}".format(acc_lr))
print("Decision Tree Classifier Accuracy: {:.2f}".format(acc_dt))
print("Random Forest Classifier Accuracy: {:.2f}".format(acc_rf))

# Print classification reports
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("Decision Tree Classifier Classification Report:")
print(classification_report(y_test, y_pred_dt))

print("Random Forest Classifier Classification Report:")
print(classification_report(y_test, y_pred_rf))
