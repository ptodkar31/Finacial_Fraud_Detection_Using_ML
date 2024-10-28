import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("C:/Users/Documents/Online_Fraud.csv")

# Remove rows with missing values
df.dropna(inplace=True)

# Convert categorical columns to numerical values using label encoding
df['type'] = LabelEncoder().fit_transform(df['type'])
df['nameOrig'] = LabelEncoder().fit_transform(df['nameOrig'])
df['nameDest'] = LabelEncoder().fit_transform(df['nameDest'])

# Define features and target
target = df['isFraud']
features = df.drop(columns=['isFraud', 'isFlaggedFraud'])

# Perform feature selection using chi-squared test
selector = SelectKBest(score_func=chi2, k=5)  # Choose number of top features (k) based on your data
selected_features = selector.fit_transform(features, target)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.20, random_state=42)

# Create a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# Define hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'max_depth': np.arange(3, 11),  # Range of max depth values from 3 to 10
    'min_samples_split': [2, 4, 6, 8],  # Range of min samples split
    'min_samples_leaf': [1, 2, 3, 4]  # Range of min samples leaf
}

# Perform RandomizedSearchCV with 20 iterations and 3-fold cross-validation
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='f1',  # Use F1-score as the evaluation metric
    random_state=42,
    n_jobs=-1  # Use parallel processing
)

# Fit the model using RandomizedSearchCV
random_search.fit(x_train, y_train)

# Get the best model from the search
best_model = random_search.best_estimator_

# Save the best model and feature selector
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(selector, 'feature_selector.pkl')

# Evaluate the model's performance on the test data
y_pred = best_model.predict(x_test)
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.2f}")

# Print the classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the best hyperparameters found
print("Best Hyperparameters:")
print(random_search.best_params_)
