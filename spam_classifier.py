import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("spam.csv")

X = data['email']
y = data['label']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

model = LogisticRegression()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf)

print("Accuracy:", scores.mean())
