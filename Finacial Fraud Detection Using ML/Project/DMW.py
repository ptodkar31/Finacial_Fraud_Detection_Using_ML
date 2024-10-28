from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Initialize Flask application
app = Flask(__name__)

# Train the model and feature selector when the application starts
def train_model_and_selector():
    # Load your dataset
    df = pd.read_csv("C:/Users/Documents/Online_Fraud.csv")

    # Preprocess the dataset
    df = df.dropna()  # Remove rows with any missing values
    df['type'] = LabelEncoder().fit_transform(df['type'])
    df['nameOrig'] = LabelEncoder().fit_transform(df['nameOrig'])
    df['nameDest'] = LabelEncoder().fit_transform(df['nameDest'])

    # Define features and target
    features = df.drop(columns=['isFraud', 'isFlaggedFraud'])
    target = df['isFraud']

    # Perform feature selection using chi-squared test
    k = 5  # Choose number of top features (k) based on your data
    selector = SelectKBest(score_func=chi2, k=k)
    selected_features = selector.fit_transform(features, target)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.20, random_state=42)

    # Create and train the Decision Tree Classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Evaluate the model's performance on the test data
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score: {f1:.2f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, selector

# Train the model and feature selector
model, feature_selector = train_model_and_selector()

# Define route for home page
@app.route('/')
def index():
    return render_template('index1.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        step = int(request.form['step'])
        type_val = int(request.form['type'])
        amount = float(request.form['amount'])
        Name_Origin = int(request.form['nameorg'])
        oldbalanceOrg = float(request.form['oldorg'])
        newbalanceOrig = float(request.form['neworg'])
        Name_Dest = int(request.form['namedest'])
        oldbalancedest = float(request.form['olddest'])
        newbalancedest = float(request.form['newdest'])


        # Convert input data into a numpy array
        input_data = np.array([[step,type_val, amount, Name_Origin,oldbalanceOrg, newbalanceOrig,Name_Dest,oldbalancedest,newbalancedest]])

        # Preprocess input data using the feature selector
        input_data_transformed = feature_selector.transform(input_data)

        # Make a prediction using the trained model
        prediction = model.predict(input_data_transformed)

        # Determine the result ('Fraud' or 'No Fraud')
        result = 'Fraud' if prediction[0] == 1 else 'No Fraud'
        return render_template('result1.html', result=result)

    except Exception as e:
        return render_template('error1.html', error=str(e))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
