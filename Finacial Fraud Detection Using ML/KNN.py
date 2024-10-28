import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

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

# Create a Logistic Regression Classifier
model = LogisticRegression(random_state=42)

# Define hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'C': np.logspace(-4, 4, 10),  # Regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],  # Solver types
}

# Perform RandomizedSearchCV with 20 iterations and 3-fold cross-validation
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='accuracy',  # Use accuracy as the evaluation metric
    random_state=42,
    n_jobs=-1  # Use parallel processing
)

# Fit the model using RandomizedSearchCV
random_search.fit(x_train, y_train)

# Get the best model from the search
best_model = random_search.best_estimator_ 

# Evaluate the model's performance on the test data
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the best hyperparameters found
print("Best Hyperparameters:")
print(random_search.best_params_)

# Predict new data points
new_data_points = [
    [1, 2, 20128, 1118430673, 20128, 0, 339924917, 6268, 12145.85],
    [1, 4, 181.0, 165236, 181.0, 0.0, 73550, 0.0, 0.0],
    [1, 2, 181.0, 961662, 181.0, 0.0, 65464, 21182.0, 0.0],
    [1, 2, 35063.63, 1635772897, 35063063, 0, 1983025922, 31140, 7550.03]
]

# Convert new data points to a pandas DataFrame
new_data_df = pd.DataFrame(new_data_points, columns=features.columns)

# Transform new data points using the feature selector
new_data_points_transformed = selector.transform(new_data_df)

# Predict the class of each new data point
predictions = best_model.predict(new_data_points_transformed)

# Print predictions for each new data point
for index, prediction in enumerate(predictions, start=1):
    print(f"Prediction for data point {index}: {prediction}")
