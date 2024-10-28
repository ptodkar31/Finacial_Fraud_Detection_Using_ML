
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("C:/Users/Documents/Online_Fraud.csv")


df.dropna(inplace=True)


df['type'] = LabelEncoder().fit_transform(df['type'])
df['nameOrig'] = LabelEncoder().fit_transform(df['nameOrig'])
df['nameDest'] = LabelEncoder().fit_transform(df['nameDest'])


target = df['isFraud']
features = df.drop(columns=['isFraud', 'isFlaggedFraud'])

selector = SelectKBest(score_func=chi2, k=5)  
selected_features = selector.fit_transform(features, target)


x_train, x_test, y_train, y_test = train_test_split(selected_features, target, test_size=0.20, random_state=42)


model = DecisionTreeClassifier(random_state=42)

param_dist = {
    'max_depth': np.arange(3, 11),  
    'min_samples_split': [2, 4, 6, 8],  
    'min_samples_leaf': [1, 2, 3, 4]  
}


random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='f1',  
    random_state=42,
    n_jobs=-1  
)


random_search.fit(x_train, y_train)


best_model = random_search.best_estimator_


y_pred = best_model.predict(x_test)
f1 = f1_score(y_test, y_pred)
print(f"F1-score: {f1:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("Best Hyperparameters:")
print(random_search.best_params_)


new_data_points = [
    [1,2, 20128, 1118430673, 20128, 0, 339924917, 6268, 12145.85],
    [1,4,181.0,165236,181.0,0.0,73550,0.0,0.0],
    [1,2,181.0,961662,181.0,0.0,65464,21182.0,0.0],
    [1,2,35063.63,1635772897,35063063,0,1983025922,31140,7550.03]
]


new_data_df = pd.DataFrame(new_data_points, columns=features.columns)


new_data_points_transformed = selector.transform(new_data_df)


predictions = best_model.predict(new_data_points_transformed)


for index, prediction in enumerate(predictions, start=1):
    print(f"Prediction for data point {index}: {prediction}")
    

accuracy = (y_pred == y_test).mean() * 100
print(f"Accuracy: {accuracy:.2f}%")

