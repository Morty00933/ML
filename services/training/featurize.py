import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC = None  # будет инференситься из df

MODELS = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=200, random_state=42)
}

PARAM_GRID = {
    "logreg": {"clf__C": [0.1, 1.0, 10.0]},
    "rf": {"clf__n_estimators": [100, 200, 400], "clf__max_depth": [None, 5, 10]},
}


def build_pipeline(df: pd.DataFrame, target: str, model_key: str):
    global NUMERIC
    NUMERIC = [c for c in df.columns if c != target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC)
    ])
    clf = MODELS[model_key]
    pipe = Pipeline([
        ("prep", pre),
        ("clf", clf)
    ])
    return pipe
