import html
import re
import ssl
from typing import Iterable, List, Optional, Sequence, Union

ssl._create_default_https_context = ssl._create_unverified_context

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

NEGATIONS = {
    "not", "no", "never", "cannot", "can't", "won't", "isn't", "aren't",
    "wasn't", "weren't", "don't", "doesn't", "didn't", "hardly"
}


def ensure_nltk_resources() -> None:
    """Download commonly used NLTK resources if they are missing."""
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ]
    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif resource == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab")
            elif resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif resource == "omw-1.4":
                nltk.data.find("corpora/omw-1.4")
            elif resource == "averaged_perceptron_tagger_eng":
                nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            nltk.download(resource, quiet=True)


ensure_nltk_resources()


def remove_html_tags(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"<.*?>", " ", text)


def remove_urls(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " ", text)


def normalize_whitespace(text: str) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_characters(text: str) -> str:
    if text is None:
        return ""
    text = html.unescape(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return normalize_whitespace(text)


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True, use_stemmer: bool = False) -> str:
    if text is None:
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens: List[str] = []
    for token in tokens:
        if token in NEGATIONS:
            cleaned_tokens.append(token)
            continue
        if remove_stopwords and token in stop_words:
            continue
        if use_stemmer:
            token = token
        token = lemmatizer.lemmatize(token, pos="v")
        cleaned_tokens.append(token)
    return " ".join(cleaned_tokens)


def preprocess_text(text: Optional[str], remove_stopwords: bool = True, use_lemmatize: bool = True) -> str:
    """Apply the same cleaning pipeline used before model training."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = remove_html_tags(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = remove_emails(cleaned)
    cleaned = remove_special_characters(cleaned)
    cleaned = normalize_whitespace(cleaned)
    if use_lemmatize:
        cleaned = tokenize_and_lemmatize(cleaned, remove_stopwords=remove_stopwords, use_stemmer=False)
    else:
        tokenized = word_tokenize(cleaned.lower())
        stop_words = set(stopwords.words("english"))
        cleaned_tokens = []
        for token in tokenized:
            if token in NEGATIONS:
                cleaned_tokens.append(token)
            elif not (remove_stopwords and token in stop_words):
                cleaned_tokens.append(token)
        cleaned = " ".join(cleaned_tokens)
    return normalize_whitespace(cleaned)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for consistent text preprocessing."""

    def __init__(self, remove_stopwords: bool = True, use_lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.use_lemmatize = use_lemmatize

    def fit(self, X: Sequence[str], y: Optional[Iterable] = None):
        return self

    def transform(self, X: Sequence[str]):
        if X is None:
            return []
        return [
            preprocess_text(text, remove_stopwords=self.remove_stopwords, use_lemmatize=self.use_lemmatize)
            for text in X
        ]

    def fit_transform(self, X: Sequence[str], y: Optional[Iterable] = None, **fit_params):
        self.fit(X, y)
        return self.transform(X)


def prepare_combined_text(df, title_col: str = "title", text_col: str = "text") -> str:
    df = df.copy()
    df["combined_text"] = (
        df[title_col].fillna("").astype(str).str.strip() + " " + df[text_col].fillna("").astype(str).str.strip()
    )
    return df
