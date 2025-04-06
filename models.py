from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve, auc
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('depression_anxiety_data.csv')

for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].mean(), inplace=True)

for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

scaler = StandardScaler()
numerical_features = ['school_year','age', 'bmi', 'phq_score', 'gad_score','epworth_score']  
data[numerical_features] = scaler.fit_transform(data[numerical_features])

label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])

data['anxiety_severity'] = pd.Categorical(data['anxiety_severity'], 
                                           categories=['0','None-minimal', 'Mild', 'Moderate', 'Severe'], 
                                           ordered=True)
data['anxiety_severity'] = data['anxiety_severity'].cat.codes

depression_features = ['age', 'gender', 'bmi', 'phq_score', 'anxiety_severity','epworth_score','gad_score']
X_depression = data[depression_features]
y_depression = data['depression_severity']


X_train, X_test, y_train, y_test = train_test_split(X_depression, y_depression, test_size=0.2, random_state=42)

# Listing of models 
models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Neural Network": MLPClassifier(max_iter=1000)
}

# Evaluating here each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")