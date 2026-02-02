import streamlit as st
import pickle
import re

# ==============================
# Load trained model & vectorizer
# ==============================
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ==============================
# Text cleaning function
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.write("Enter any news text and check whether it is **REAL** or **FAKE**.")

st.success("Model loaded successfully!")

# ==============================
# User input
# ==============================
news_text = st.text_area(
    "📝 Enter News Text Here:",
    height=150,
    placeholder="Type or paste news content here..."
)

# ==============================
# Prediction
# ==============================
if st.button("🔍 Check News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text!")
    else:
        cleaned_text = clean_text(news_text)
        vector_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("👨‍💻 **Project by Harsh Saxena**")
