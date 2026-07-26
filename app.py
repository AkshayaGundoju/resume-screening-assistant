import streamlit as st
import joblib
import pandas as pd
from pypdf import PdfReader
import docx
import nltk

from src.preprocess import preprocess

st.set_page_config(
    page_title="Resume Screening Assistant",
    page_icon="📄",
    layout="centered",
)

@st.cache_resource
def ensure_nltk():
    for pkg in [
        "stopwords",
        "wordnet",
        "omw-1.4",
        "punkt",
        "punkt_tab",
    ]:
        nltk.download(pkg, quiet=True)

ensure_nltk()

# ---------- load model once ----------
@st.cache_resource
def load_artifact():
    return joblib.load("models/resume_classifier.joblib")


# ---------- file readers ----------
def read_pdf(file):
    reader = PdfReader(file)
    return " ".join((page.extract_text() or "") for page in reader.pages)


def read_docx(file):
    document = docx.Document(file)
    return " ".join(p.text for p in document.paragraphs)


def extract_text(uploaded):
    if uploaded.name.endswith(".pdf"):
        return read_pdf(uploaded)

    if uploaded.name.endswith(".docx"):
        return read_docx(uploaded)

    return uploaded.read().decode("utf-8", errors="ignore")


# ---------- UI ----------
st.title("📄 Resume Screening Assistant")
st.caption(
    "Upload a resume and the model predicts the most likely job category."
)

artifact = load_artifact()

pipeline = artifact["pipeline"]
label_encoder = artifact["label_encoder"]

with st.sidebar:
    st.header("About")
    st.write(f"Categories: **{len(label_encoder.classes_)}**")
    st.write(f"Test macro-F1: **{artifact['macro_f1']:.3f}**")
    st.write("TF-IDF + Logistic Regression, trained on ~960 resumes.")

uploaded = st.file_uploader(
    "Upload a resume",
    type=["pdf", "docx", "txt"],
)

if uploaded is not None:

    raw = extract_text(uploaded)

    if len(raw.strip()) < 50:
        st.error(
            "Could not read enough text. Is this a scanned image PDF?"
        )
        st.stop()

    cleaned = preprocess(raw)

    pred_idx = pipeline.predict([cleaned])[0]

    proba = pipeline.predict_proba([cleaned])[0]

    category = label_encoder.inverse_transform([pred_idx])[0]

    st.success(f"### Predicted category: {category}")

    st.metric("Confidence", f"{proba[pred_idx]:.1%}")

    top5 = pd.DataFrame(
        {
            "Category": label_encoder.classes_,
            "Probability": proba,
        }
    ).sort_values(
        "Probability",
        ascending=False,
    ).head(5)

    st.subheader("Top 5 Candidates")

    st.bar_chart(top5.set_index("Category"))

    with st.expander("Show Extracted Text"):
        st.text(raw[:3000])