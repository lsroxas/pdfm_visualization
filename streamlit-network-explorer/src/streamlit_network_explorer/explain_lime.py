# src/streamlit_network_explorer/explain_lime.py
from __future__ import annotations
from typing import List, Optional, Callable, Sequence
import pickle
import pathlib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

def load_pickle_model(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p.resolve()}")
    with p.open("rb") as f:
        return pickle.load(f)

def _detect_mode(model) -> str:
    """
    Heuristic: classification if predict_proba exists; else regression if predict returns floats.
    """
    if hasattr(model, "predict_proba"):
        return "classification"
    return "regression"

def make_predict_fn(model) -> Callable[[np.ndarray], np.ndarray]:
    """
    Returns a function f(X_raw) -> probabilities or predictions.
    Works whether `model` is a plain estimator or a Pipeline (it will call model.predict_proba
    if available, else model.predict).
    """
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)
    return lambda X: np.atleast_2d(model.predict(X)).reshape(-1, 1)

def build_explainer(
    background_df: pd.DataFrame,
    feature_names: Sequence[str],
    class_names: Optional[Sequence[str]] = None,
    categorical_feature_names: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
) -> LimeTabularExplainer:
    """
    Create a LimeTabularExplainer using raw feature space (no transforms).
    background_df: a representative sample of your data distribution
    feature_names: columns in the raw input order
    categorical_feature_names: list of feature names that are categorical (optional)
    mode: "classification" or "regression" (auto-detected if None)
    """
    X_bg = background_df[feature_names].to_numpy()

    categorical_features = None
    if categorical_feature_names:
        name_to_index = {n: i for i, n in enumerate(feature_names)}
        categorical_features = [name_to_index[n] for n in categorical_feature_names if n in name_to_index]

    explainer = LimeTabularExplainer(
        training_data=X_bg,
        feature_names=list(feature_names),
        class_names=list(class_names) if class_names is not None else None,
        categorical_features=categorical_features,
        verbose=False,
        mode=mode or "classification",
        discretize_continuous=True,
        random_state=42,
    )
    return explainer

def explain_row(
    model,
    explainer: LimeTabularExplainer,
    row_df: pd.DataFrame,                 # single-row DataFrame in raw feature space
    feature_names: Sequence[str],
    num_features: int = 10,
    top_labels: int = 1,
):
    """
    Run LIME for one row. Returns (exp, label_index) where:
      - exp is a lime.explanation.Explanation
      - label_index is the class index explained (for classification)
    """
    # prediction function taking raw features
    predict_fn = make_predict_fn(model)

    x = row_df[feature_names].iloc[0].to_numpy()
    # For classification, we pick the model’s predicted class to explain
    label_to_explain = None
    if hasattr(model, "predict_proba"):
        probs = predict_fn(row_df[feature_names].to_numpy())
        label_to_explain = int(np.argmax(probs, axis=1)[0])

    exp = explainer.explain_instance(
        data_row=x,
        predict_fn=lambda X: predict_fn(pd.DataFrame(X, columns=feature_names)),
        num_features=num_features,
        top_labels=top_labels,
        labels=[label_to_explain] if label_to_explain is not None else None,
    )
    return exp, label_to_explain