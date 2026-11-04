"""Risultati fold per fold (un fold = un soggetto di test) del modello migliore: XGB, tutte le feature."""
import pandas as pd
from sklearn.metrics import confusion_matrix

from pipeline import CLASSIFIERS, FEATURE_SETS, TASKS, build_table, loso_predictions, per_subject_scores

table = build_table().dropna(subset=FEATURE_SETS["All"]).reset_index(drop=True)
X, groups = table[FEATURE_SETS["All"]].to_numpy(), table["subject"].to_numpy()

for task, mapping in TASKS.items():
    y, classes = table["phase"].map(mapping).factorize(sort=True)
    pred = loso_predictions(X, y, groups, *CLASSIFIERS["XGB"])
    scores = per_subject_scores(y, pred, groups)
    # matrice di confusione di ogni soggetto, appiattita: "vero>predetto" = numero di finestre
    pairs = [f"{t}>{p}" for t in classes for p in classes]
    confusion = pd.DataFrame({s: confusion_matrix(y[groups == s], pred[groups == s], labels=range(len(classes))).ravel()
                              for s in scores.index}, index=pairs).T
    print(f"\n=== {task} (classi: {list(classes)})")
    print(scores.join(confusion).sort_values("f1").round(2).rename(index=lambda s: s[3:11]).to_string())
