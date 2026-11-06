# VitaStress — reproduction

Riproduzione della pipeline di classificazione dello stress del paper:

> Schreiber, P., Burbach, S., Cinar, B., Mackert, L., Maleshkova, M. (2025).
> *VitaStress: A Multimodal Dataset for Stress Detection.* ICMI '25, pp. 673–681.
> https://doi.org/10.1145/3716553.3754337

Progetto di tirocinio, Università degli Studi dell'Insubria.

## Contenuto

Tutto è in [`pipeline.ipynb`](pipeline.ipynb):

1. **Tabella delle finestre**: segnali del bracciale Corsano CardioWatch ricampionati a finestre di 30 s,
   etichettati con le 4 fasi (baseline, physical, cognitive, public) da `*_annotation.csv` e normalizzati
   con z-score per soggetto.
2. **Feature** (come nel paper):
   - fisiologiche: heart rate (`bpm`), conduttanza cutanea (`stress_skin`), temperatura cutanea (`temp_sk1`);
   - movimento: media, deviazione standard e varianza dell'accelerometro sui 3 assi.
3. **Classificazione**: binaria (stress vs baseline) e ternaria (baseline / physical / stress) con
   Random Forest, LDA, k-NN, Decision Tree, Logistic Regression e XGBoost. Validazione leave-one-subject-out
   con grid search annidata, metriche accuracy e F1.
4. **Approfondimento Random Forest**: risultati per soggetto e importanza delle feature.

## Problemi del dataset gestiti

- `a5e823ad`: la baseline annotata dura molto più di ~10 min. Si tengono al massimo gli ultimi 10 min.
- `7bb4dafd`: `Cognitive Stop` senza timestamp, quindi il soggetto non ha la fase cognitive.
- `ddca342b`: assi Y e Z dell'accelerometro di segno opposto (bracciale indossato capovolto).
  La correzione è opzionale (`FLIPPED_AXES`).

## Uso

1. Scarica il dataset da https://github.com/paulvincenz/VitaStress
2. La prima cella del notebook installa le dipendenze (numpy, pandas, scikit-learn, xgboost).
3. Nella cella successiva del notebook imposta `DATA_DIR` sul percorso della cartella `data/` del dataset.
   Gli output vengono salvati in `OUT_DIR` (default `analysis/`, creata in automatico).

## Licenza

Il dataset VitaStress è distribuito con licenza **CC BY-NC 4.0** e non è incluso in questa repo.
Se usi il dataset, cita gli autori del paper.
