import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords (one time)
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# =========================
# Text Cleaning Function (GLOBAL)
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection System")
st.write("Enter any news text and check whether it is **REAL or FAKE**.")

# =========================
# Load & Train Model (CACHED SAFE WAY)
# =========================
@st.cache_resource
def train_model():
    fake = pd.read_csv("dataset/Fake.csv")
    true = pd.read_csv("dataset/True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)

    # Reduce size for speed
    data = data.sample(8000).reset_index(drop=True)

    data["clean_text"] = data["text"].apply(clean_text)

    X = data["clean_text"]
    y = data["label"]

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, vectorizer

with st.spinner("Training model, please wait..."):
    model, vectorizer = train_model()

st.success("✅ Model loaded successfully!")

# =========================
# User Input Section
# =========================
news_input = st.text_area("📝 Enter News Text Here:")

if st.button("🔍 Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        cleaned = clean_text(news_input)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")

st.markdown("---")
st.markdown("👨‍💻 **Project by Harsh Saxena**")
