"""Replica della pipeline del preprint VitaStress (Figura 3) e della classificazione (sez. 5)."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent  # cartella Tirocinio/
DATA_DIR = ROOT / "VitaStress" / "data"
OUT_DIR = ROOT / "analysis"

# --- scelte della pipeline (il paper non le specifica)
# finestre a cavallo della fine di una fase:
#   "full"  -> scartate, si tengono solo le finestre interamente dentro la fase
#   "start" -> si tengono quelle che iniziano dentro la fase (conteggi coerenti con la Fig. 8)
LABEL_RULE = "start"
# su quali finestre di ogni soggetto si calcolano media e std dello z-score:
#   "phases" -> le finestre delle 4 fasi
#   "experiment" -> dalla prima all'ultima annotation;
#   "recording" -> tutta la registrazione
ZSCORE_BASE = "phases"
# assi dell'accelerometro da invertire per i soggetti col braccialetto ruotato di 180° attorno a X.
# Vuoto = dati come pubblicati (riproduzione del paper); correzione: {"ddca342b": ["accY", "accZ"]}
FLIPPED_AXES = {"ddca342b": ["accY", "accZ"]}
#FLIPPED_AXES = {}

WINDOW = pd.Timedelta("30s")  # "resampled to have a sampling frequency of 1/30 seconds" (sez. 3.6)

# (Start, Stop) di ogni fase in *_annotation.csv; "public" è solo il discorso, senza preparazione
PHASE_LABELS = {
    "baseline": ("Baseline Start (Start of Experiment)", "Baseline Stop"),
    "cognitive": ("Cognitive: Start", "Cognitive: Stop"),
    "physical": ("Physical: Start", "Physical: Stop"),
    "public": ("Public Speaking Start", "Public Speaking Stop"),
}

# file -> segnale vitale (già processato da Corsano), da ricampionare a 30 s
VITALS = {"activity.csv": "bpm", "emography.csv": "stress_skin", "temperature.csv": "temp_sk1"}
ZERO_IS_MISSING = {"bpm", "temp_sk1"}  # qui uno 0 indica un valore mancante

PHYSIO = list(VITALS.values())
MOTION = [f"acc{axis}_{stat}" for axis in "XYZ" for stat in ("mean", "std", "var")]
FEATURE_SETS = {"All": PHYSIO + MOTION, "Physio": PHYSIO, "Motion": MOTION}

# fase -> classe, per i due compiti del paper (sez. 5)
TASKS = {
    "binary": {"baseline": "baseline", "physical": "stress", "cognitive": "stress", "public": "stress"},
    "ternary": {"baseline": "baseline", "physical": "physical",
                "cognitive": "cognitive+public", "public": "cognitive+public"},
}

# i 6 classificatori del paper (sez. 3.6) con la griglia di iperparametri della grid search
# (il paper non riporta le griglie: queste sono piccole e scelte da noi)
CLASSIFIERS = {
    "RF": (RandomForestClassifier(random_state=42), {"n_estimators": [100, 300], "max_depth": [None, 10]}),
    "LDA": (LinearDiscriminantAnalysis(solver="lsqr"), {"shrinkage": [None, "auto"]}),
    "kNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 9, 15]}),
    "DT": (DecisionTreeClassifier(random_state=42), {"max_depth": [3, 5, 10, None]}),
    "LR": (LogisticRegression(max_iter=1000), {"C": [0.01, 0.1, 1, 10]}),
    "XGB": (XGBClassifier(random_state=42, n_jobs=1), {"n_estimators": [100, 300], "max_depth": [3, 6]}),
}


def read_csv(path, columns, time_column="date"):
    df = pd.read_csv(path, usecols=[time_column, *columns])
    df[time_column] = pd.to_datetime(df[time_column], utc=True, format="ISO8601")  # precisione variabile
    return df.dropna(subset=[time_column]).set_index(time_column)


def phase_intervals(annotation_path):
    """{fase: (start, stop)}: ultimo Stop e ultimo Start prima di esso (gestisce i riavvii)."""
    buttons = read_csv(annotation_path, ["Button Name"], "timestamp")["Button Name"]
    intervals = {}
    for phase, (start_label, stop_label) in PHASE_LABELS.items():
        stop = buttons.index[buttons == stop_label].max()
        starts = buttons.index[(buttons == start_label) & (buttons.index < stop)]
        if starts.empty:  # anche quando manca lo Stop (stop = NaT)
            print(f"  {annotation_path.parent.name}: fase {phase} senza Start/Stop")
            continue
        start = starts.max()
        # id_a5e823ad: "Baseline Start" premuto ~1 h prima; le altre baseline durano ~10 min
        if phase == "baseline" and stop - start > pd.Timedelta("12min"):
            start = stop - pd.Timedelta("10min")
        intervals[phase] = (start, stop)
    return intervals, buttons.index.min(), buttons.index.max()


def subject_windows(files):
    """Una riga per finestra da 30 s: accelerometro aggregato + segnali vitali ricampionati."""
    acc = read_csv(files["acc.csv"], ["accX", "accY", "accZ"]).resample(WINDOW).agg(["mean", "std", "var"])
    acc.columns = [f"{col}_{stat}" for col, stat in acc.columns]  # ("accX", "mean") -> "accX_mean"

    columns = [acc]
    for file, name in VITALS.items():
        signal = read_csv(files[file], [name])[name]
        if name in ZERO_IS_MISSING:
            signal = signal.replace(0, np.nan)
        signal = signal.resample(WINDOW).mean()
        if name == "stress_skin":  # 1 valore/min: riempie la finestra vuota tra due valori
            signal = signal.interpolate(limit=1, limit_area="inside")
        columns.append(signal)
    return pd.concat(columns, axis=1, sort=True)  # sorgenti con estensione temporale diversa


def subject_table(subject_dir, label_rule, zscore_base):
    """Finestre di un soggetto che cadono in una fase, z-scorate e con la colonna "phase"."""
    # "<uuid>_temperature.csv" -> "temperature.csv" (non confonde heat_flux_sensor_temperature)
    files = {f.name.split("_", 1)[1]: f for f in subject_dir.iterdir() if "_" in f.name}
    windows = subject_windows(files)
    for subject, axes in FLIPPED_AXES.items():
        if subject in subject_dir.name:  # invertire il segno cambia solo la media, non std e var
            windows[[f"{axis}_mean" for axis in axes]] *= -1
    intervals, first, last = phase_intervals(files["annotation.csv"])

    start_times, end_times = windows.index, windows.index + WINDOW
    phase = pd.Series(np.nan, index=windows.index, dtype=object)
    for name, (start, stop) in intervals.items():
        ends_inside = end_times <= stop if label_rule == "full" else start_times < stop
        phase[(start_times >= start) & ends_inside] = name

    base = {"phases": phase.notna(),
            "experiment": (start_times >= first) & (end_times <= last),
            "recording": np.ones(len(windows), dtype=bool)}[zscore_base]
    z = (windows - windows[base].mean()) / windows[base].std(ddof=0)
    return z[phase.notna()].assign(phase=phase, subject=subject_dir.name)


def build_table(label_rule=LABEL_RULE, zscore_base=ZSCORE_BASE):
    """Tabella di tutti i soggetti: una riga per finestra etichettata."""
    subject_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    tables = (subject_table(d, label_rule, zscore_base) for d in subject_dirs)
    return pd.concat(tables).rename_axis("window_start").reset_index()


def loso_predictions(X, y, groups, classifier, grid):
    """LOSO annidata: per ogni soggetto di test, la grid search sceglie gli iperparametri con
    una LOSO interna sui soli soggetti di training; il modello migliore, riaddestrato su
    tutti i soggetti di training, predice le finestre del soggetto di test."""
    pred = np.empty_like(y)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        search = GridSearchCV(classifier, grid, scoring="f1_macro", cv=LeaveOneGroupOut(), n_jobs=-1)
        pred[test] = search.fit(X[train], y[train], groups=groups[train]).predict(X[test])
    return pred


def per_subject_scores(y, pred, groups):
    """Accuracy e macro-F1 di ogni soggetto, cioè di ogni fold della LOSO."""
    df = pd.DataFrame({"y": y, "pred": pred, "subject": groups})
    return df.groupby("subject").apply(lambda d: pd.Series({
        "acc": accuracy_score(d["y"], d["pred"]), "f1": f1_score(d["y"], d["pred"], average="macro")}))


def evaluate_all(table):
    """Una riga per ogni (compito, feature set, classificatore) con media e std sui fold."""
    # LDA, kNN e LR non accettano NaN: si scartano le finestre con almeno una feature mancante
    table = table.dropna(subset=FEATURE_SETS["All"]).reset_index(drop=True)
    rows = []
    for task, mapping in TASKS.items():
        y = table["phase"].map(mapping).factorize(sort=True)[0]  # classi -> 0..n-1 (per XGBoost)
        for set_name, features in FEATURE_SETS.items():
            X = table[features].to_numpy()
            for clf_name, (classifier, grid) in CLASSIFIERS.items():
                groups = table["subject"].to_numpy()
                scores = per_subject_scores(y, loso_predictions(X, y, groups, classifier, grid), groups)
                rows.append({"task": task, "features": set_name, "classifier": clf_name,
                             "acc": scores["acc"].mean(), "acc_std": scores["acc"].std(ddof=0),
                             "f1": scores["f1"].mean(), "f1_std": scores["f1"].std(ddof=0)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = build_table()
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"windows_{LABEL_RULE}_{ZSCORE_BASE}.csv"
    table.to_csv(out_path, index=False)
    print(f"salvata {out_path.name}: {len(table)} finestre, per fase:", table.groupby("phase").size().to_dict())
    print(f"senza NaN: {len(table.dropna(subset=FEATURE_SETS['All']))} finestre "
          "(paper, Fig. 8: 357 baseline + 757 stress)")

    results = evaluate_all(table)
    for metric in ("acc", "f1"):
        results[f"{metric}_text"] = results[metric].map("{:.2f}".format) + " ± " + results[f"{metric}_std"].map("{:.2f}".format)

    # stesso formato delle Tabelle 2-5 del paper: righe = feature set, colonne = classificatori
    for task in TASKS:
        for metric in ("acc", "f1"):
            print(f"\n{task} - {metric}")
            print(results[results["task"] == task].pivot(index="features", columns="classifier", values=f"{metric}_text")
                  .loc[list(FEATURE_SETS), list(CLASSIFIERS)].to_string())
