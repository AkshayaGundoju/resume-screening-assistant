import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_resume(text):
    """
    Cleans resume text for machine learning.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\+?\d[\d\s\-]{8,}\d", " ", text)

    # Remove everything except letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)



from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def normalize(text: str) -> str:
    """Tokenise, drop stop words and short tokens, lemmatise."""

    tokens = word_tokenize(text)

    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS
        and len(tok) > 2
        and not tok.isdigit()
    ]

    return " ".join(tokens)


def preprocess(text: str) -> str:
    """The full text pipeline: clean, then normalise."""
    return normalize(clean_resume(text))