import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load your data
data = pd.read_csv('iBeacon_RSSI_Labeled.csv')

features = ['date', 'b3001', 'b3002', 'b3003', 'b3004', 'b3005', 'b3006', 'b3007', 'b3008', 'b3009', 'b3010', 'b3011', 'b3012', 'b3013']
target_variables = ['location']

# Create a new DataFrame without missing values
data_no_missing = data.dropna(subset=target_variables)

X = data_no_missing[features]
y = data_no_missing[target_variables]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing for the numerical features
numerical_features = features[1:]  # exclude 'date' if it's not numerical
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define the preprocessing for the categorical features
categorical_features = ['date']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=1000, random_state=42))
])

# Train the classifier
pipeline.fit(X_train, y_train.values.ravel())

# Save the trained model
model_path = 'path_to_saved_model/randomforest_model.joblib'
os.makedirs('path_to_saved_model', exist_ok=True)
joblib.dump(pipeline, model_path)
