import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import sys
from vectorizer_module import BERTVectorizer

def main():
    print("Starting autism text classification training...")
    
    # Load data
    try:
        print("Loading dataset...")
        df = pd.read_csv("data/txt_autism_dataset_enhanced.csv")
        if 'text' not in df.columns or 'Autism_Diagnosis' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'Autism_Diagnosis' columns")
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Initialize BERT
    try:
        print("🔄 Initializing BERT vectorizer...")
        vectorizer = BERTVectorizer()
    except Exception as e:
        print(f"❌ BERT initialization failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Create embeddings
    print("⚙️ Creating embeddings (this may take a few minutes)...")
    try:
        X = vectorizer.transform(df['text'])
        y = df['Autism_Diagnosis'].values
    except Exception as e:
        print(f"❌ Embedding failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("🤖 Training classifier...")
    try:
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            C=0.01,  # Stronger regularization
            penalty='l2'
        )
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ Training failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n📊 Evaluation Metrics:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\n📝 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    try:
        joblib.dump(model, "models/autism_model.pkl")
        vectorizer.save("models/bert_vectorizer.pkl")
        print("\n✅ Model saved successfully to models/ directory")
    except Exception as e:
        print(f"❌ Failed to save model: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()