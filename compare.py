"""Esegue pipeline.py con tutte le 6 combinazioni di LABEL_RULE e ZSCORE_BASE e indica la migliore."""
from itertools import product

import pandas as pd

from pipeline import OUT_DIR, build_table, evaluate_all

all_results = []
for label_rule, zscore_base in product(["full", "start"], ["phases", "experiment", "recording"]):
    print(f"--- {label_rule} / {zscore_base}", flush=True)
    results = evaluate_all(build_table(label_rule, zscore_base))
    all_results.append(results.assign(label_rule=label_rule, zscore_base=zscore_base))
all_results = pd.concat(all_results)
OUT_DIR.mkdir(exist_ok=True)
all_results.to_csv(OUT_DIR / "compare_configs.csv", index=False)

config = ["label_rule", "zscore_base"]

# macro-F1 medio sui 18 modelli (3 feature set x 6 classificatori) di ogni compito:
# premia la combinazione che funziona bene in generale, non con un solo modello fortunato
summary = all_results.pivot_table(index=config, columns="task", values="f1", aggfunc="mean")
summary["mean"] = summary.mean(axis=1)
summary = summary.sort_values("mean", ascending=False)
print("\nmacro-F1 medio su tutti i modelli")
print(summary.round(3).to_string())

# miglior singolo modello di ogni combinazione, per compito (il dato riportato dal paper)
best =(all_results.sort_values("f1", ascending=False).drop_duplicates([*config, "task"])
        .set_index([*config, "task"])[["classifier", "features", "acc", "f1"]].sort_index())
print("\nmiglior modello per combinazione")
print(best.round(3).to_string())

label_rule, zscore_base = summary.index[0]
print(f"\nCombinazione migliore: LABEL_RULE = \"{label_rule}\", ZSCORE_BASE = \"{zscore_base}\" "
      f"(macro-F1 medio {summary['mean'].iloc[0]:.3f})")
