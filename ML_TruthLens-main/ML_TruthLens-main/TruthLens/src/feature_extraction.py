from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 20000,
    ngram_range: tuple = (1, 2),
    min_df: float = 2,
    max_df: float = 0.95,
    stop_words: str = "english",
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        strip_accents="unicode",
        sublinear_tf=True,
    )


def get_default_vectorizer_config() -> Dict[str, Any]:
    return {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "stop_words": "english",
    }
