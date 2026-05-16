import streamlit as st
import joblib
import pandas as pd
import re
from bs4 import BeautifulSoup
import spacy

# PAGE CONFIG
st.set_page_config(
    page_title="Review Classification System",
    page_icon="⭐",
    layout="wide"
)

# LOAD SPACY
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_lg")

nlp = load_spacy()

# PREPROCESS FUNCTION
def preprocess(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\s+', ' ', text).strip()

    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)

# LOAD MODELS
@st.cache_resource
def load_models():
    models = {
        "SVC": joblib.load("models/svc.joblib"),
        "Logistic Regression": joblib.load("models/logistic_reg.joblib"),
        "MultinomialNB": joblib.load("models/multinomial.joblib"),
        "Random Forest": joblib.load("models/random_forest.joblib"),
        "SGD Classifier": joblib.load("models/sgd_classifier.joblib")
    }

    vectorizer = joblib.load("models/vectorizer.joblib")

    return models, vectorizer

models, vectorizer = load_models()

# TITLE
st.title("Review Classification Dashboard")
st.markdown("Analyze review authenticity using 5 Machine Learning Models")

# INPUT
review = st.text_area(
    "Enter Review Text:",
    height=200,
    placeholder="Type review here..."
)

# PREDICT BUTTON
if st.button("Process", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter review text.")
    else:
        cleaned_text = preprocess(review)

        x = vectorizer.transform([cleaned_text])

        results = []

        fake_count = 0
        genuine_count = 0

        for model_name, model in models.items():

            pred = model.predict(x)[0]

            label = "Fake" if pred == 0 else "Genuine"

            if label == "Fake":
                fake_count += 1
            else:
                genuine_count += 1

            results.append({
                "Model": model_name,
                "Prediction": label
            })

        df = pd.DataFrame(results)

        # display table
        st.subheader("Model Predictions")
        st.dataframe(df, use_container_width=True)

        # final voting
        st.subheader("Final Voting Result")

        if fake_count > genuine_count:
            st.error("Final Decision: Fake Review")
        else:
            st.success("Final Decision: Genuine Review")

# FOOTER
st.markdown("---")
st.caption("Coursework Project | Review Classification using 5 ML Models")