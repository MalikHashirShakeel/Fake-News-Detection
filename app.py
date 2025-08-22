import streamlit as st
import joblib
import re
import nltk
import spacy
from nltk.corpus import stopwords

# ======================
# Load Required Assets
# ======================
# Load TF-IDF Vectorizer & Model
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("random_forest_model.pkl")

# Load NLP resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ======================
# Preprocessing Function
# ======================
def preprocess_text(text):
    try:
        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuation, digits, and extra spaces
        text = re.sub(r'[^a-z\s]', '', text)

        # 3. Tokenize + remove stopwords
        tokens = [word for word in text.split() if word not in stop_words]

        # 4. Lemmatization with spaCy
        doc = nlp(" ".join(tokens))
        tokens = [token.lemma_ for token in doc]

        return " ".join(tokens)
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return ""

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #ffffff;
            font-family: 'Courier New', monospace;
        }
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #ff4b82;
            margin-bottom: 20px;
        }
        .result-box {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 1.3rem;
            font-weight: bold;
            margin-top: 20px;
        }
        .real {
            background-color: #0f5132;
            color: #d1e7dd;
        }
        .fake {
            background-color: #842029;
            color: #f8d7da;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.write("Enter the **title** and **content** of a news article below. The app will preprocess the text and classify it as **Real** ✅ or **Fake** ❌.")

# ======================
# Input Section
# ======================
title = st.text_input("Enter News Title:")
content = st.text_area("Enter News Content:", height=200)

if st.button("🔍 Classify"):
    if not title.strip() and not content.strip():
        st.warning("⚠️ Please provide either a title, content, or both.")
    else:
        try:
            # Combine title + content
            full_text = title + " " + content
            processed_text = preprocess_text(full_text)

            if processed_text.strip() == "":
                st.error("⚠️ Preprocessing returned empty text. Please enter valid input.")
            else:
                # TF-IDF Transformation
                vectorized = tfidf.transform([processed_text])

                # Prediction
                pred = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]

                # Output Result
                if pred == 1:
                    st.markdown(
                        f"<div class='result-box real'>✅ This news is classified as **REAL**<br>Confidence: {proba[1]*100:.2f}%</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='result-box fake'>❌ This news is classified as **FAKE**<br>Confidence: {proba[0]*100:.2f}%</div>",
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
