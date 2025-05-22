import json
from pathlib import Path
import pandas as pd
from perturber import DatasetPreparer
from anomaly_det import AnomalyDetector
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score


# ------------------------------------------------------------------------
# Helper per “flattenare” una confusion-matrix 2×2 in 4 numeri
# (TN, FP, FN, TP) così Excel resta compatto
# ------------------------------------------------------------------------
def cm_flat(cm_2x2):
    tn, fp, fn, tp = cm_2x2.ravel()
    return tn, fp, fn, tp

# ------------------------------------------------------------------------
# Core: una singola iterazione su UN dataset, con 3 modalità di run
# ------------------------------------------------------------------------
def one_dataset_run(df_source,
                    env_cols,
                    ind_cols,
                    run_id,
                    type:str = "perturb"):
    """
    Restituisce un dizionario con tutte le metriche di una iterazione,
    inclusa la Recall sugli outlier, per:
      a) dati grezzi (standard),
      b) residui contestuali (contextual),
      c) solo indicatori (subset ind_cols).
    """
    if type == "perturb":
        # 1) preprocessing
        prep = DatasetPreparer(
            env_cols=env_cols,
            ind_cols=ind_cols,
            random_state=run_id
        )
        data_proc, _ = prep.prepare(df_source.copy())



    elif type == "synthetic":
        data_proc = df_source.copy()

    # 2) Separazione labels da input
    y_true_anom = data_proc["is_anomaly"]
    y_true_out = data_proc["is_outlier"]
    data_proc.drop(columns=["is_anomaly", "is_outlier"], inplace=True)

    # 3) anomaly detector
    detector = AnomalyDetector(
        env_cols=env_cols,
        ind_cols=ind_cols,
        max_feature=1,
        n_estimator=300,
        random_state=run_id,
        bayesian_iter=1,
        contamination=0.2,
    )

    # --- a) IsolationForest su dati grezzi -----------------------------
    df_iso_std    = detector.find_anomalies(data_proc.copy())
    y_pred_std    = df_iso_std["anomaly"]
    acc_std_anom   = accuracy_score(y_true_anom, y_pred_std)
    cm_std_anom    = confusion_matrix(y_true_anom, y_pred_std)
    cm_std_out     = confusion_matrix(y_true_out, y_pred_std)
    recall_std_out = recall_score(y_true_out, y_pred_std)

    # --- b) IsolationForest su residui (contextual) --------------------
    df_iso_ctx     = detector.find_contextual_anomalies(data_proc.copy())
    y_pred_ctx     = df_iso_ctx["contextual_anomaly"]
    acc_ctx_anom   = accuracy_score(y_true_anom, y_pred_ctx)
    cm_ctx_anom    = confusion_matrix(y_true_anom, y_pred_ctx)
    cm_ctx_out     = confusion_matrix(y_true_out, y_pred_ctx)
    recall_ctx_out = recall_score(y_true_out, y_pred_ctx)

    # --- c) IsolationForest solo su indicatori (ind_cols) -------------
    df_ind         = data_proc[ind_cols].copy()
    df_iso_ind     = detector.find_anomalies(df_ind)
    y_pred_ind     = df_iso_ind["anomaly"]
    acc_ind_anom    = accuracy_score(y_true_anom, y_pred_ind)
    cm_ind_anom     = confusion_matrix(y_true_anom, y_pred_ind)
    cm_ind_out      = confusion_matrix(y_true_out, y_pred_ind)
    recall_ind_out  = recall_score(y_true_out, y_pred_ind)

    # 4) flatten delle confusion-matrix e ritorno record
    tn_sa, fp_sa, fn_sa, tp_sa = cm_flat(cm_std_anom)
    tn_so, fp_so, fn_so, tp_so = cm_flat(cm_std_out)
    tn_ca, fp_ca, fn_ca, tp_ca = cm_flat(cm_ctx_anom)
    tn_co, fp_co, fn_co, tp_co = cm_flat(cm_ctx_out)
    tn_ia, fp_ia, fn_ia, tp_ia = cm_flat(cm_ind_anom)
    tn_io, fp_io, fn_io, tp_io = cm_flat(cm_ind_out)

    return {
        "Run": run_id,

        # Standard su anomaly
        "acc_std_iso_anomaly":      acc_std_anom,
        "TN_sa": tn_sa, "FP_sa": fp_sa, "FN_sa": fn_sa, "TP_sa": tp_sa,

        # Standard su outlier + Recall(outlier)
        "recall_std_iso_outlier":   recall_std_out,
        "TN_so": tn_so, "FP_so": fp_so, "FN_so": fn_so, "TP_so": tp_so,

        # Contextual su anomaly
        "acc_ctx_iso_anomaly":      acc_ctx_anom,
        "TN_ca": tn_ca, "FP_ca": fp_ca, "FN_ca": fn_ca, "TP_ca": tp_ca,

        # Contextual su outlier + Recall(outlier)
        "recall_ctx_iso_outlier":   recall_ctx_out,
        "TN_co": tn_co, "FP_co": fp_co, "FN_co": fn_co, "TP_co": tp_co,

        # Indicator-only su anomaly
        "acc_ind_iso_anomaly":      acc_ind_anom,
        "TN_ia": tn_ia, "FP_ia": fp_ia, "FN_ia": fn_ia, "TP_ia": tp_ia,

        # Indicator-only su outlier + Recall(outlier)
        "recall_ind_iso_outlier":   recall_ind_out,
        "TN_io": tn_io, "FP_io": fp_io, "FN_io": fn_io, "TP_io": tp_io,
    }

# ------------------------------------------------------------------------
# Main: legge il JSON e crea un Excel per ciascun dataset
# ------------------------------------------------------------------------
def main(json_config_path: str, n_runs: int = 10):
    with open(json_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for ds_cfg in config.get("datasets"):
        name     = ds_cfg["name"]
        env_cols = ds_cfg["env_cols"]
        ind_cols = ds_cfg["ind_cols"]

        if "data_path" in ds_cfg:
            df_raw = pd.read_csv(ds_cfg["data_path"])
        else:
            raise ValueError(f"Dataset '{name}' non ha data_path")

        df_raw = df_raw.dropna().reset_index(drop=True)

        records = []
        for run in range(n_runs):
            print(f"[{name}] run {run+1}/{n_runs}")
            rec = one_dataset_run(df_raw, env_cols, ind_cols, run, type="synthetic")
            records.append(rec)

        df_results = pd.DataFrame(records)
        out_path = Path("results_2") / f"{name}.xlsx"
        df_results.to_excel(out_path, index=False)
        print(f" --> salvato in {out_path.resolve()}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python run_anomaly_batches.py <config.json>")
        sys.exit(1)
    main(sys.argv[1], n_runs=10)


