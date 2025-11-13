from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib
import warnings
import sys
import pandas as pd  

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: Required packages not installed properly.", file=sys.stderr)
    print("Please run these commands:", file=sys.stderr)
    print("pip uninstall -y sentence-transformers transformers huggingface-hub", file=sys.stderr)
    print("pip install sentence-transformers==2.2.2 huggingface-hub==0.13.4 transformers==4.26.1", file=sys.stderr)
    sys.exit(1)

class BERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Failed to initialize SentenceTransformer: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            if isinstance(X, pd.Series):
                X = X.tolist()
            elif isinstance(X, str):
                X = [X]
            return np.array(self.model.encode(X, show_progress_bar=False))
        except Exception as e:
            print(f"Text embedding failed: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    def save(self, path):
        joblib.dump({'model_name': self.model_name}, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        return cls(model_name=data['model_name'])
