import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# ---------------- TRAINING DATA ----------------
data = {
    "resume": [
        "python machine learning tensorflow data science sql pandas numpy",
        "html css javascript frontend backend web developer",
        "java spring boot microservices backend sql",
        "data analysis machine learning python pandas statistics",
        "ui ux design figma photoshop creativity wireframe"
    ],
    "role": [
        "ML Engineer",
        "Web Developer",
        "Java Developer",
        "Data Scientist",
        "UI/UX Designer"
    ]
}

df = pd.DataFrame(data)

# ---------------- VECTORIZE ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume"])

# ---------------- TRAIN MODEL ----------------
model = LogisticRegression()
model.fit(X, df["role"])

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")