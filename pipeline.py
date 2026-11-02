import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


""" FUNCTIONS AND CONSTANTS """

DATA_DIR = "VitaStress/data/"
WINDOW = "30s"  # "resampled to have a sampling frequency of 1/30 seconds" (preprint, sez. 3.6)
# OUT_PATH_START = "analysis/windows_30s_start.csv"  # finestre di bordo incluse, baseline tagliata (non normalizzata)
PATH = "z-score_filtered"
# baseline più lunghe di BASELINE_MAX vengono ridotte agli ultimi BASELINE_KEEP prima di
# "Baseline Stop" (protocollo: ~10 min; tutte 9.8-10.2 min tranne id_a5e823ad, vedi label_windows)
BASELINE_MAX = pd.Timedelta("12min")
BASELINE_KEEP = pd.Timedelta("10min")

# etichette Start/Stop per fase, osservate in *_annotation.csv (non uniformi tra loro,
# "public" usa deliberatamente solo il discorso, non la preparazione)
PHASE_LABELS = {
    "baseline": ("Baseline Start (Start of Experiment)", "Baseline Stop"),
    "cognitive": ("Cognitive: Start", "Cognitive: Stop"),
    "physical": ("Physical: Start", "Physical: Stop"),
    "public": ("Public Speaking Start", "Public Speaking Stop"),
}
PHASES = list(PHASE_LABELS.keys())  # solo i nomi delle fasi: baseline, cognitive, physical, public

# schema della pipeline (preprint, Figura 3): l'accelerometro (grezzo, ~32 Hz) viene
# AGGREGATO per finestra con mean/std/var, i segnali vitali (già processati da Corsano)
# vengono solo RICAMPIONATI sulla stessa griglia
ACC_SOURCE = ("_acc.csv", ["accX", "accY", "accZ"])
VITAL_SOURCES = {
    "_activity.csv": ["bpm"],
    "_emography.csv": ["stress_skin"],  # EDA / skin conductance
    "_temperature.csv": ["temp_sk1"],  # la "skin temperature" del paper (in realtà CBT); temp_sk2 è sempre 0
}

# colonne in cui un valore mancante è scritto come 0 invece che lasciato vuoto
ZERO_IS_MISSING = {"bpm", "temp_sk1"}

def subject_files(subject_dir):
    """Path di tutti i file del soggetto, indicizzati per suffisso esatto:
    "<uuid>_temperature.csv" -> "_temperature.csv". Gli UUID non contengono "_",
    quindi il suffisso inizia al primo "_". Con endswith() invece
    "_heat_flux_sensor_temperature.csv" verrebbe preso anche per "_temperature.csv"."""
    with os.scandir(subject_dir) as files:
        return {f.name[f.name.find("_"):]: f.path for f in files}


def get_phase_intervals(annotation_csv_path):
    """Per ciascuna fase: (ultimo Start prima dell'ultimo Stop, ultimo Stop).
    Gestisce riavvii/duplicati nelle annotation. Ritorna None per una fase
    se manca lo Start o lo Stop corrispondente."""
    ann = pd.read_csv(annotation_csv_path)
    # format="ISO8601": alcune righe non hanno i microsecondi, un formato fisso
    # inferito dalla prima riga fallisce sulle righe con precisione diversa
    ann["timestamp"] = pd.to_datetime(ann["timestamp"], utc=True, format="ISO8601")

    intervals = {}
    for phase, (start_label, stop_label) in PHASE_LABELS.items():
        stops = ann.loc[ann["Button Name"] == stop_label, "timestamp"]
        if stops.empty:
            intervals[phase] = None
            continue
        stop = stops.max()

        starts = ann.loc[(ann["Button Name"] == start_label) & (ann["timestamp"] < stop), "timestamp"]
        if starts.empty:
            intervals[phase] = None
            continue
        start = starts.max()

        intervals[phase] = (start, stop)
    return intervals


def read_signal(path, columns):
    """Colonne richieste di un CSV, indicizzate per data, con gli zeri-mancanti a NaN."""
    df = pd.read_csv(path, usecols=["date", *columns])
    df["date"] = pd.to_datetime(df["date"], utc=True, format="ISO8601")
    for col in columns:
        if col in ZERO_IS_MISSING:
            df[col] = df[col].replace(0, np.nan)
    return df.set_index("date")


def resample_vital(signal):
    """Ricampiona un segnale vitale sulla griglia da WINDOW. Se il segnale è più lento
    della finestra (emography: 1 valore/min) il ricampionamento lascia finestre vuote
    tra un valore e l'altro: riempiamo solo quei buchi per interpolazione lineare, non i
    mancanti veri (nel paper stress_skin ha ~0% di mancanti, Figura 4)."""
    resampled = signal.resample(WINDOW).mean()
    native_step = signal.index.to_series().diff().median()
    holes = int(native_step / pd.Timedelta(WINDOW)) - 1
    if holes > 0:
        resampled = resampled.interpolate(method="linear", limit=holes, limit_area="inside")
    return resampled


def subject_windows(paths):
    """Una riga per finestra da WINDOW con tutte le feature del soggetto affiancate
    (allineamento: tutte le sorgenti sono sulla stessa griglia temporale)."""
    suffix, columns = ACC_SOURCE
    acc = read_signal(paths[suffix], columns).resample(WINDOW).agg(["mean", "std", "var"])
    acc.columns = [f"{col}_{stat}" for col, stat in acc.columns]  # ("accX", "mean") -> "accX_mean"

    signals = [acc]
    for suffix, columns in VITAL_SOURCES.items():
        signals.append(resample_vital(read_signal(paths[suffix], columns)))
    print("START SUBJECT WINDOW")
    print(pd.concat(signals, axis=1, sort=True))
    print("END SUBJECT WINDOW")
    return pd.concat(signals, axis=1, sort=True)


def subject_windows_zscore(paths):
    """Come subject_windows, ma ogni feature è già z-scorata per soggetto usando TUTTA
    la registrazione (anche briefing, preparazione, pause, recovery), non solo le 4 fasi.
    Le statistiche si calcolano prima di label_windows, che poi tiene solo le finestre
    nelle fasi."""
    windows = subject_windows(paths)
    return (windows - windows.mean()) / windows.std(ddof=0)


def experiment_span(annotation_csv_path):
    """(prima annotation, ultima annotation) del soggetto. Le righe senza timestamp
    (es. l'ultimo "Cognitive Stop" di id_7bb4dafd) vengono ignorate."""
    ann = pd.read_csv(annotation_csv_path)
    timestamps = pd.to_datetime(ann["timestamp"], utc=True, format="ISO8601").dropna()
    return timestamps.min(), timestamps.max()


def subject_windows_zscore_filtered(paths):
    """Come subject_windows_zscore, ma media e std di ogni feature si calcolano solo
    sulle finestre interamente comprese tra la prima e l'ultima annotation, escludendo
    quello che è stato registrato prima e dopo l'esperimento. Il risultato contiene
    comunque tutte le finestre: è label_windows a tenere poi solo quelle nelle fasi."""
    windows = subject_windows(paths)
    start, stop = experiment_span(paths["_annotation.csv"])
    inside = (windows.index >= start) & (windows.index + pd.Timedelta(WINDOW) <= stop)
    experiment = windows[inside]
    return (windows - experiment.mean()) / experiment.std(ddof=0)


def label_windows(windows, intervals, rule="full", trim_baseline=False):
    """Tiene solo le finestre che appartengono a una delle 4 fasi e aggiunge la colonna
    "phase". L'indice è l'inizio della finestra. La griglia da 30 s è allineata
    all'orologio, gli Start/Stop no: rule decide cosa fare con le finestre a cavallo
    del confine di una fase.
    - "full":  solo finestre interamente dentro la fase (le finestre di bordo si scartano)
    - "start": finestre che INIZIANO dentro la fase, cioè l'etichetta che riceve ogni
               riga del resample in base al suo timestamp. Con questa regola i campioni di
               stress tornano con la Figura 8 del preprint (774 contro 757 del paper,
               che probabilmente ha scartato anche le righe con NaN)
    Con trim_baseline=True una baseline più lunga di BASELINE_MAX viene ridotta agli
    ultimi BASELINE_KEEP: in id_a5e823ad "Baseline Start" è premuto ~1 h prima
    dell'inizio della registrazione."""
    labeled = []
    for phase, interval in intervals.items():
        if interval is None:
            continue
        start, stop = interval
        if trim_baseline and phase == "baseline" and stop - start > BASELINE_MAX:
            start = stop - BASELINE_KEEP

        if rule == "full":
            inside = (windows.index >= start) & (windows.index + pd.Timedelta(WINDOW) <= stop)
        elif rule == "start":
            inside = (windows.index >= start) & (windows.index < stop)
        else:
            raise ValueError(f"rule sconosciuta: {rule}")
        labeled.append(windows[inside].assign(phase=phase))
    return pd.concat(labeled)


def build_window_table(windows_fn=subject_windows, label_rule="full", trim_baseline=False):
    """Tabella di tutti i soggetti: una riga per finestra etichettata, colonne
    feature + "phase" + "subject". Le feature sono quelle restituite da windows_fn
    (subject_windows: NON normalizzate); label_rule e trim_baseline vanno a
    label_windows. Ritorna anche la lista (soggetto, motivo) di cosa è stato scartato."""
    tables = []
    skipped = []

    with os.scandir(DATA_DIR) as entries:
        subject_dirs = sorted((e for e in entries if e.is_dir()), key=lambda e: e.name)

    for subject_dir in subject_dirs:
        paths = subject_files(subject_dir)

        missing = [s for s in ["_annotation.csv", ACC_SOURCE[0], *VITAL_SOURCES] if s not in paths]
        if missing:
            skipped.append((subject_dir.name, f"file mancanti: {missing}"))
            continue

        intervals = get_phase_intervals(paths["_annotation.csv"])
        for phase, interval in intervals.items():
            if interval is None:
                skipped.append((subject_dir.name, f"fase {phase} senza Start/Stop"))

        print("INTERVALS: %s", intervals)
        windows = label_windows(windows_fn(paths), intervals, label_rule, trim_baseline)
        print("WINDOWS: %s", windows)
        tables.append(windows.assign(subject=subject_dir.name))

    return pd.concat(tables), skipped


def build_window_table_zscore():
    """Come build_window_table, ma con le feature z-scorate su tutta la registrazione
    di ciascun soggetto (vedi subject_windows_zscore)."""
    return build_window_table(subject_windows_zscore)


def build_window_table_zscore_filtered():
    """Come build_window_table, ma con le feature z-scorate sull'intervallo tra prima
    e ultima annotation di ciascun soggetto (vedi subject_windows_zscore_filtered)."""
    return build_window_table(subject_windows_zscore_filtered)


""" FEATURES EXTRACTION """
# Passi 1-4 della pipeline (preprint, Figura 3): estrazione feature, ricampionamento /
# aggregazione a 30 s, allineamento, etichette. La tabella viene salvata PRIMA della
# normalizzazione per soggetto, così si può cambiare quella senza rileggere i CSV.
import os
import pandas as pd

if PATH == "z-score_filtered":
    path_out = "analysis/windows_30s_normalized_filtered.csv"
elif PATH == "z-score":
    path_out = "analysis/windows_30s_normalized.csv"
else: path_out = "analysis/windows_30s.csv"

table, skipped = build_window_table()

# l'inizio della finestra diventa una colonna normale; "phase" tiene le 4 fasi originali,
# le etichette binaria/ternaria si ricavano dopo con una mappatura
table = table.rename_axis("window_start").reset_index()
feature_columns = [col for col in table.columns if col not in ("subject", "phase", "window_start")]
table = table[["subject", "phase", "window_start", *feature_columns]]

os.makedirs(os.path.dirname(path_out), exist_ok=True)
table.to_csv(path_out)
# table.to_pickle(OUT_PATH) # per versione in .pkl


if skipped:
    print("Scartati:")
    for name, reason in skipped:
        print(f"  - {name}: {reason}")
print(f"salvata {path_out}: {table.shape[0]} finestre x {len(feature_columns)} feature")
print("soggetti:", table["subject"].nunique())
print("finestre per fase:", table.groupby("phase").size().to_dict())
# confronto con la Figura 4 del preprint (% mancanti per classe)
print("\n% mancanti per fase:")
print((table.groupby("phase")[feature_columns].apply(lambda g: g.isna().mean()) * 100).round(1).T)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# insiemi di feature del paper (sez. 5: "All", "Physio", "Motion")
PHYSIO_FEATURES = ["bpm", "stress_skin", "temp_sk1"]
MOTION_FEATURES = [f"acc{axis}_{stat}" for axis in "XYZ" for stat in ("mean", "std", "var")]
FEATURE_SETS = {
    "All": PHYSIO_FEATURES + MOTION_FEATURES,
    "Physio": PHYSIO_FEATURES,
    "Motion": MOTION_FEATURES,
}

# fase originale -> classe, per i due compiti del paper (sez. 5)
TASKS = {
    "binary": {"baseline": "baseline", "physical": "stress", "cognitive": "stress", "public": "stress"},
    "ternary": {"baseline": "baseline", "physical": "physical",
                "cognitive": "cognitive+public", "public": "cognitive+public"},
}


def zscore_per_subject(table, columns):
    """z-score di ogni colonna separatamente per ciascun soggetto, calcolato sulle
    righe delle 4 fasi ("normalized to account for inter-personal differences")."""
    by_subject = table.groupby("subject")[columns]
    normalized = table.copy()
    normalized[columns] = (table[columns] - by_subject.transform("mean")) / by_subject.transform("std", ddof=0)
    return normalized


def get_classifiers():
    # i 6 classificatori del paper (sez. 3.6); iperparametri fissi, il paper fa grid search
    return {
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'kNN (k=9)': KNeighborsClassifier(n_neighbors=9),  # nel paper WESAD viene utilizzato k=9
        'DT': DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=5, random_state=42),
        'LR': LogisticRegression(max_iter=1000, C=1.0),
        'XGB': XGBClassifier(random_state=42),
    }


def evaluate_loso(X, y, groups, class_names):
    """Leave-one-subject-out: un fold per soggetto. Accuracy e macro-F1 sono calcolate
    per fold e poi mediate (media ± std sui fold, come nelle Tabelle 2-5 del paper);
    la matrice di confusione somma le predizioni di tutti i fold."""
    results = {}
    print(f'{"":<10} {"acc":>13} {"macro-F1":>13}')

    for name, clf in get_classifiers().items():
        accs, f1s = [], []
        y_true_all, y_pred_all = [], []

        for tr, te in LeaveOneGroupOut().split(X, y, groups=groups):
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])

            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average='macro', zero_division=0))
            y_true_all.extend(y[te])
            y_pred_all.extend(pred)

        results[name] = {
            'acc': np.mean(accs), 'acc_std': np.std(accs),
            'f1': np.mean(f1s), 'f1_std': np.std(f1s),
            'cm': confusion_matrix(y_true_all, y_pred_all, labels=range(len(class_names))),
        }
        print(f'{name:<10} {np.mean(accs):>6.2f} ± {np.std(accs):.2f} {np.mean(f1s):>6.2f} ± {np.std(f1s):.2f}')

    best = max(results, key=lambda k: results[k]['f1'])
    print(f'Migliore (F1): {best}  F1={results[best]["f1"]:.2f} ± {results[best]["f1_std"]:.2f}')
    print('matrice di confusione (righe = vero, colonne = predetto):', list(class_names))
    print(results[best]['cm'])
    return results, best

""" CLASSIFIERS """
# base dello z-score per soggetto:
# "phases"    -> tabella grezza, z-score calcolato qui sulle sole finestre delle 4 fasi
# "recording" -> tabella già z-scorata su tutta la registrazione (build_window_table_zscore)
# "experiment" -> tabella già z-scorata tra prima e ultima annotation (build_window_table_zscore_filtered)
ZSCORE_BASE = "phases"
# tabella grezza usata con "phases": OUT_PATH (solo finestre intere) oppure OUT_PATH_START
RAW_PATH = path_out

feature_columns = FEATURE_SETS["All"]
if ZSCORE_BASE == "phases":
    table = pd.read_csv(RAW_PATH, parse_dates=["window_start"])
    table_z = zscore_per_subject(table, feature_columns)
elif ZSCORE_BASE == "recording":
    table_z = pd.read_csv(OUT_PATH_ZSCORE, parse_dates=["window_start"])
else:
    table_z = pd.read_csv(OUT_PATH_ZSCORE_FILTERED, parse_dates=["window_start"])

# LDA, kNN e LR non accettano NaN: si scartano le finestre con almeno una feature mancante
complete = table_z[feature_columns].notna().all(axis=1)
print(f"finestre scartate per NaN: {(~complete).sum()} su {len(table_z)}")
table_z = table_z[complete].reset_index(drop=True)
groups = table_z["subject"].to_numpy()

all_results = {}
for task, mapping in TASKS.items():
    # classi -> interi 0..n-1 (XGBoost li richiede); class_names[i] è il nome della classe i
    y, class_names = pd.factorize(table_z["phase"].map(mapping), sort=True)
    for set_name, columns in FEATURE_SETS.items():
        print(f"\n=== {task} / {set_name} ({len(columns)} feature)")
        X = table_z[columns].to_numpy()
        all_results[(task, set_name)], _ = evaluate_loso(X, y, groups, class_names)

# riepilogo nello stesso formato delle Tabelle 2-5 del paper (righe = feature set, colonne = classificatori)
for task in TASKS:
    for metric in ("acc", "f1"):
        summary = pd.DataFrame({set_name: {clf: r[metric] for clf, r in all_results[(task, set_name)].items()}
                                for set_name in FEATURE_SETS}).T
        print(f"\n{task} - {metric}")
        print(summary.round(2))