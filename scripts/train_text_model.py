import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import sys
from vectorizer_module import BERTVectorizer
import matplotlib.pyplot as plt

def load_data(filepath):
    """Load data with validation"""
    try:
        df = pd.read_csv(filepath)
        if df.isnull().values.any():
            df = df.dropna()
        if 'text' not in df.columns or 'Autism_Diagnosis' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'Autism_Diagnosis' columns")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    # Load data
    df = load_data("data/txt_autism_dataset_enhanced.csv")
    
    # Initialize vectorizer
    try:
        print("Initializing BERT vectorizer...")
        vectorizer = BERTVectorizer()
    except Exception as e:
        print(f"Failed to initialize BERT: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Create embeddings
    print("Creating embeddings...")
    try:
        X = vectorizer.transform(df['text'])
        y = df['Autism_Diagnosis'].values
    except Exception as e:
        print(f"Embedding failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize model
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=0.1,
        solver='liblinear'
    )
    
    # Cross-validation
    print("Running cross-validation...")
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f"\nCross-Validation ROC AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    except Exception as e:
        print(f"Cross-validation failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training classifier...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Training failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Evaluation:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # Save model
    try:
        joblib.dump(model, "models/autism_model.pkl")
        vectorizer.save("models/bert_vectorizer.pkl")
        print("\nModel saved successfully!")
    except Exception as e:
        print(f"Failed to save model: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()