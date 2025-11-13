import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

def preprocess_and_train_model():
    df = pd.read_csv('./data/Autism-Adult-Data-Cleaned.csv')
    
    drop_cols = ['result', 'country_of_residence', 'ethnicity', 'age_desc', 'relation', 'autism', 'used_app_before']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    df['Autism_Diagnosis'] = df['Autism_Diagnosis'].str.strip().map({'YES': 1, 'NO': 0})
    categorical_cols = df.select_dtypes(include='object').columns.drop('Autism_Diagnosis', errors='ignore')
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    X = df.drop(columns=['Autism_Diagnosis'])
    y = df['Autism_Diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    imputer = SimpleImputer(strategy='most_frequent')
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=X.columns)
    
    # Transform test data (using fitted imputer)
    X_test_imputed = imputer.transform(X_test)
    X_test_imputed = pd.DataFrame(X_test_imputed, columns=X.columns)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        penalty='l2',
        C=0.1,
        solver='liblinear',
        random_state=42
    )
    lr.fit(X_train_scaled, y_train)
    
    y_pred = lr.predict(X_test_scaled)
    print("\nLogistic Regression:")
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Cross-validation
    cv_results = cross_validate(
        lr, X_train_scaled, y_train, 
        cv=5, scoring=['accuracy', 'precision', 'recall', 'f1']
    )
    print("\nCV Results:")
    print("Accuracy:", np.mean(cv_results['test_accuracy']))
    print("Precision:", np.mean(cv_results['test_precision']))
    print("Recall:", np.mean(cv_results['test_recall']))
    print("F1:", np.mean(cv_results['test_f1']))
    
   
    os.makedirs('./models', exist_ok=True)
    joblib.dump(lr, './models/logistic_model.pkl')
    joblib.dump(scaler, './models/scaler.pkl')
    print("✅ Logistic Regression model and scaler saved successfully!")
    
    # Print feature importance from Logistic Regression
    feature_weights = pd.Series(lr.coef_[0], index=X.columns)
    print("\nFeature Weights (Logistic Regression):")
    print(feature_weights.sort_values(key=abs, ascending=False)) 


if __name__ == "__main__":
    preprocess_and_train_model()
