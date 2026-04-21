import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Extra columns remove (Kaggle dataset me hoti hain)
data = data[['v1', 'v2']]

# Rename columns (easy understanding ke liye)
data.columns = ['label', 'message']

# Features & target
X = data['message']
y = data['label']

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Label Encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Model
model = LogisticRegression()

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf)

print("Accuracy for each fold:", scores)
print("Average Accuracy:", scores.mean())
