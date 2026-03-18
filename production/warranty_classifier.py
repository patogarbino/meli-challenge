"""
warranty_classifier.py

Instalación:
    pip install sentence-transformers scikit-learn pandas

Uso:
    from warranty_classifier import classify_warranties

    # Un solo df
    df_result = classify_warranties(df_labeled, df_full)

    # Varios dfs
    df_a, df_b, df_c = classify_warranties(df_labeled, df_a, df_b, df_c)
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

MERGE_MAP = {
    "12+ meses": "7–12 meses",
    "3–5 años": "5+ años",
}


def classify_warranties(
    df_labeled: pd.DataFrame,
    *dfs: pd.DataFrame,
    warranty_col: str = "warranty",
    label_col: str = "warranty_label",
    text_col: str = "warranty",
    label_out: str = "warranty_label",
    confidence_out: str = "warranty_confidence",
    merge_map: dict = MERGE_MAP,
    evaluate: bool = True,
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Entrena sobre df_labeled y predice sobre uno o más DataFrames.

    Parámetros
    ----------
    df_labeled     : DataFrame etiquetado para entrenar.
    *dfs           : Uno o más DataFrames a predecir.
    warranty_col   : Columna de texto en los dfs a predecir.
    label_col      : Columna de etiquetas en df_labeled.
    text_col       : Columna de texto en df_labeled.
    label_out      : Columna de salida con la categoría predicha.
    confidence_out : Columna de salida con la confianza (0–1).
    merge_map      : Fusiones de clases con pocos ejemplos.
    evaluate       : Si True imprime F1 macro con cross-validation.

    Retorna
    -------
    Si se pasa un solo df  → devuelve ese df con las columnas nuevas.
    Si se pasan varios dfs → devuelve una lista de dfs en el mismo orden.
    """
    model, clf, le = _train(
        df_labeled, label_col, text_col, merge_map, evaluate
    )
    results = [
        _predict(df, model, clf, le, warranty_col, label_out, confidence_out)
        for df in dfs
    ]
    return results[0] if len(results) == 1 else results


# ── Privadas ───────────────────────────────────────────────────────────────────


def _train(df_labeled, label_col, text_col, merge_map, evaluate):
    df = df_labeled.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].replace(merge_map)

    print(f"[train] {len(df)} ejemplos | {df[label_col].nunique()} clases")

    model = SentenceTransformer(MODEL_NAME)
    X = model.encode(
        df[text_col].fillna("").astype(str).tolist(),
        show_progress_bar=True,
        batch_size=64,
    )

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])

    clf = SGDClassifier(
        loss="modified_huber",
        alpha=0.001,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X, y)

    if evaluate:
        scores = cross_val_score(clf, X, y, cv=5, scoring="f1_macro")
        print(
            f"[train] F1 macro (5-fold): {scores.mean():.3f} ± {scores.std():.3f}"
        )

    return model, clf, le


def _predict(df, model, clf, le, warranty_col, label_out, confidence_out):
    df = df.copy()
    mask = df[warranty_col].notna() & (df[warranty_col].str.strip() != "")

    df[label_out] = "Sin dato"
    df[confidence_out] = 0.0

    if mask.sum() == 0:
        return df

    X = model.encode(
        df.loc[mask, warranty_col].astype(str).tolist(),
        show_progress_bar=False,
        batch_size=64,
    )
    preds = le.inverse_transform(clf.predict(X))
    probas = clf.predict_proba(X).max(axis=1)

    df.loc[mask, label_out] = preds
    df.loc[mask, confidence_out] = probas.round(3)

    print(f"[predict] {len(df)} filas | confianza media: {probas.mean():.3f}")
    return df
