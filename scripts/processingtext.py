import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from vectorizer_module import BasicVectorizer  # Using simplified vectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df = pd.read_csv("data/txt_autism_dataset.csv")
df['clean_text'] = df['text'].apply(clean_text)

vectorizer = BasicVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['Autism_Diagnosis']

joblib.dump(vectorizer, "vectorizer.pkl")
print("Preprocessing complete!")