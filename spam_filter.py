"""
SMS spam filter: training, evaluation and prediction.

The model is a scikit-learn pipeline (TF-IDF vectorizer + linear Support Vector
Classifier) trained on the UCI SMS Spam Collection.

Run this file to retrain the model, print an evaluation report and save the
fitted model to ``model/spam_filter.joblib``:

    python spam_filter.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Resources" / "SMSSpamCollection.csv"
MODEL_PATH = ROOT / "model" / "spam_filter.joblib"

TEST_SIZE = 0.33
RANDOM_STATE = 42
LABELS = ["ham", "spam"]


def load_data(path=DATA_PATH):
    """Load the dataset as a DataFrame with 'label' and 'text_message' columns."""
    return pd.read_csv(path)


def split_data(sms_text_df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split messages and labels into train and test sets (X_train, X_test, y_train, y_test)."""
    X = sms_text_df["text_message"]
    y = sms_text_df["label"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_pipeline(stop_words=None):
    """
    Build an untrained TF-IDF + LinearSVC pipeline.

    Stop words are kept by default, as in the original Gradio app: the model
    scored slightly higher on the test set that way than with
    stop_words='english'.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words=stop_words)),
        ("clf", LinearSVC()),
    ])


def sms_classification(sms_text_df, stop_words=None):
    """
    Train the spam filter on the training split of the DataFrame.

    Parameters:
    - sms_text_df (pd.DataFrame): DataFrame containing 'text_message' and 'label' columns.
    - stop_words: passed to TfidfVectorizer (None keeps every word, 'english' removes common ones).

    Returns:
    - text_clf (Pipeline): Fitted pipeline model.
    """
    X_train, _, y_train, _ = split_data(sms_text_df)
    return build_pipeline(stop_words).fit(X_train, y_train)


def evaluate(text_clf, sms_text_df):
    """
    Score a fitted model on the held-out test split.

    Returns a dict with accuracy, the confusion matrix (rows = actual, columns =
    predicted, in ``LABELS`` order), a text classification report and a
    DataFrame of the misclassified test messages.
    """
    _, X_test, _, y_test = split_data(sms_text_df)
    y_pred = text_clf.predict(X_test)
    wrong = y_test != y_pred
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS),
        "report": classification_report(y_test, y_pred, labels=LABELS, digits=3),
        "errors": pd.DataFrame({
            "actual": y_test[wrong],
            "predicted": y_pred[wrong.to_numpy()],
            "text_message": X_test[wrong],
        }),
    }


def save_model(text_clf, path=MODEL_PATH):
    """Save a fitted model to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(text_clf, path)
    return path


def load_model(path=MODEL_PATH):
    """
    Load the saved model, or train a fresh one if no saved model exists.
    """
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    return sms_classification(load_data())


def predict_label(text_clf, text):
    """Return the raw label ('ham' or 'spam') for a single message."""
    return text_clf.predict([text])[0]


def sms_prediction(text, text_clf=None):
    """
    Predict whether a given text message is spam or not.

    Parameters:
    - text (str): The text message to be classified.
    - text_clf (Pipeline, optional): A fitted model. Defaults to the saved model.

    Returns:
    - str: A message indicating whether the text is classified as spam or not.
    """
    if text_clf is None:
        text_clf = _default_model()
    if predict_label(text_clf, text) == "ham":
        return f'The text message: "{text}", is not spam.'
    return f'The text message: "{text}", is spam.'


_model_cache = None


def _default_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache


def main():
    sms_text_df = load_data()
    text_clf = sms_classification(sms_text_df)
    results = evaluate(text_clf, sms_text_df)

    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted; ham, spam):")
    print(results["confusion_matrix"])
    print("\n" + results["report"])
    print(f"Misclassified test messages ({len(results['errors'])}):")
    for row in results["errors"].itertuples():
        print(f"  [{row.actual} -> {row.predicted}] {row.text_message}")

    path = save_model(text_clf)
    print(f"\nSaved model to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
