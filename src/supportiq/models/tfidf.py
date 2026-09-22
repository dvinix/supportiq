"""Classical baseline model: Word + Character N-Gram TF-IDF with Logistic Regression."""

from pathlib import Path
from typing import Any

import joblib
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from supportiq.core.logger import get_logger
from supportiq.evaluation.classification import compute_classification_metrics

logger = get_logger(__name__)


class TFIDFBaseline:
    """Fast classical baseline using sub-word and word TF-IDF features with Logistic Regression."""

    def __init__(
        self,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        max_word_features: int = 10000,
        max_char_features: int = 20000,
        c_param: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.random_state = random_state
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=word_ngram_range, max_features=max_word_features)),
            (
                "char",
                TfidfVectorizer(
                    ngram_range=char_ngram_range,
                    analyzer="char",
                    max_features=max_char_features,
                ),
            ),
        ])
        self.intent_clf = LogisticRegression(
            C=c_param, max_iter=1000, random_state=random_state
        )
        self.category_clf = LogisticRegression(
            C=c_param, max_iter=1000, random_state=random_state
        )
        self.is_fitted = False

    def fit(self, train_df: pl.DataFrame) -> "TFIDFBaseline":
        """Fit feature extractors and classifiers on training dataframe."""
        texts = train_df["instruction"].to_list()
        intents = train_df["intent"].to_list()
        categories = train_df["category"].to_list()

        logger.info("Extracting TF-IDF features for %d training samples...", len(texts))
        X = self.vectorizer.fit_transform(texts)

        logger.info("Fitting Intent Classifier...")
        self.intent_clf.fit(X, intents)

        logger.info("Fitting Category Classifier...")
        self.category_clf.fit(X, categories)

        self.is_fitted = True
        return self

    def predict(self, texts: list[str]) -> tuple[list[str], list[str]]:
        """Predict (category, intent) for a list of input texts."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")
        X = self.vectorizer.transform(texts)
        pred_categories = self.category_clf.predict(X).tolist()
        pred_intents = self.intent_clf.predict(X).tolist()
        return pred_categories, pred_intents

    def evaluate(self, test_df: pl.DataFrame) -> dict[str, Any]:
        """Evaluate baseline on a test split, returning accuracy and F1 metrics."""
        texts = test_df["instruction"].to_list()
        true_categories = test_df["category"].to_list()
        true_intents = test_df["intent"].to_list()

        pred_categories, pred_intents = self.predict(texts)

        intent_metrics = compute_classification_metrics(true_intents, pred_intents)
        cat_metrics = compute_classification_metrics(true_categories, pred_categories)

        return {
            "total_samples": len(texts),
            "intent": intent_metrics,
            "category": cat_metrics,
        }

    def save(self, filepath: Path | str) -> None:
        """Save model bundle to disk using joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved baseline model to %s", path)

    @classmethod
    def load(cls, filepath: Path | str) -> "TFIDFBaseline":
        """Load trained model bundle from disk."""
        return joblib.load(filepath)
