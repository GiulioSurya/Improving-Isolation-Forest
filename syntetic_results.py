"""
grid_anomaly_benchmark.py  – v2

Benchmark a griglia del modello AnomalyDetector
al variare della covarianza di X (Σₓ) e della
varianza dello *external shock* (σ²_shock).

Funzione principale
-------------------
run_accuracy_grid(
    cov_x_list: Sequence[np.ndarray | float],
    shock_var_list: Sequence[float],
    *,
    n_runs        = 5,
    n_samples     = 1_000,
    n_env         = 3,
    n_ind         = 2,
    anomaly_rate  = 0.05,
    noise_std     = 1.0,
    detector_kwargs: dict | None = None,
    random_state  = None,
) -> pd.DataFrame
    → righe = label di Σₓ,
      colonne = varianza shock,
      celle = accuracy media su n_runs simulazioni.
"""

from __future__ import annotations
import math
from typing import Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from syntetic_dataset import GenerateData          # :contentReference[oaicite:0]{index=0}
from anomaly_det import AnomalyDetector            # :contentReference[oaicite:1]{index=1}



def _as_cov_matrix(cov: Union[np.ndarray, float], n_env: int) -> np.ndarray:
    """Converte *cov* in matrice (n_env×n_env), se necessario."""
    if np.isscalar(cov):
        return np.eye(n_env) * float(cov)
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.shape != (n_env, n_env):
        raise ValueError(
            f"cov_x matrix must have shape ({n_env},{n_env}), got {cov_arr.shape}"
        )
    return cov_arr


def _cov_label(cov: Union[np.ndarray, float]) -> str:
    """Crea un'etichetta compatta e hash-able per la matrice di covarianza."""
    if np.isscalar(cov):
        return f"diag({float(cov):.2g})"
    # per evitare indici lunghissimi arrotondo a due decimali
    return np.array2string(np.asarray(cov), precision=2, separator=",", suppress_small=True)


def _single_run(
    cov_x: np.ndarray,
    shock_var: float,
    *,
    run_id: int,
    n_samples: int,
    n_env: int,
    n_ind: int,
    anomaly_rate: float,
    noise_std: float | np.ndarray,
    detector_kwargs: dict,
    base_seed: int | None,
) -> float:
    """Una singola simulazione → accuracy (Isolation Forest su dati grezzi)."""
    seed = None if base_seed is None else base_seed + run_id
    ext_std = math.sqrt(shock_var)

    # 1) dataset sintetico
    gen = GenerateData(
        n_samples=n_samples,
        n_env=n_env,
        n_ind=n_ind,
        anomaly_rate=anomaly_rate,
        noise_std=noise_std,
        cov_x=cov_x,
        ext_std=ext_std,
        random_state=seed,
    )
    df = gen.generate(return_labels=False)

    # 2) colonne e label
    env_cols = [f"env_X{i}" for i in range(n_env)]
    ind_cols = [f"ind_Y{j}" for j in range(n_ind)]
    y_true = df["is_anomaly"].to_numpy()

    # 3) detection
    detector = AnomalyDetector(
        env_cols=env_cols,
        ind_cols=ind_cols,
        max_feature=1,
        n_estimator=300,
        random_state=seed,
        **detector_kwargs,
    )

    X_raw = df.drop(columns=["is_anomaly", "is_outlier"])
    y_pred = detector.find_contextual_anomalies(X_raw)["contextual_anomaly"].to_numpy()
    #y_pred = detector.find_anomalies(X_raw)["anomaly"].to_numpy()

    return accuracy_score(y_true, y_pred)


# -----------------------------------------------------------------------
# API principale
# -----------------------------------------------------------------------
def run_accuracy_grid(
    cov_x_list: Sequence[Union[np.ndarray, float]],
    shock_var_list: Sequence[float],
    *,
    n_runs: int = 5,
    n_samples: int = 1_000,
    n_env: int = 3,
    n_ind: int = 2,
    anomaly_rate: float = 0.05,
    noise_std: float | np.ndarray = 1.0,
    detector_kwargs: dict | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Calcola l'accuracy media di AnomalyDetector su tutte le combinazioni
    (Σₓ, σ²_shock).

    * **cov_x_list**: sequenza di matrici (n_env×n_env) *oppure* scalari.
      Se un elemento è uno scalare *s*, verrà usato Σₓ = s·I.
    * **shock_var_list**: varianze (σ²) dello shock esterno.

    Ritorna un DataFrame con indici etichettati da `_cov_label`.
    """
    if detector_kwargs is None:
        detector_kwargs = {}

    # etichette leggibili per l'indice
    cov_labels = [_cov_label(cov) for cov in cov_x_list]
    df_acc = pd.DataFrame(
        np.full((len(cov_labels), len(shock_var_list)), np.nan),
        index=pd.Index(cov_labels, name="Σₓ"),
        columns=pd.Index(shock_var_list, name="Varianza_shock"),
    )

    # loop principale
    for i, cov_raw in enumerate(cov_x_list):
        cov_mat = _as_cov_matrix(cov_raw, n_env)
        for j, shock_var in enumerate(shock_var_list):
            acc_runs = [
                _single_run(
                    cov_mat,
                    shock_var,
                    run_id=r,
                    n_samples=n_samples,
                    n_env=n_env,
                    n_ind=n_ind,
                    anomaly_rate=anomaly_rate,
                    noise_std=noise_std,
                    detector_kwargs=detector_kwargs,
                    base_seed=random_state,
                )
                for r in range(n_runs)
            ]
            df_acc.iat[i, j] = np.mean(acc_runs)

    return df_acc

if __name__ == "__main__":


    # 3 variabili di contesto
    cov_diag1 = np.eye(3) * 1.0  # Σₓ = diag(1,1,1)
    cov_diag2 = np.eye(3) * 3.0  # Σₓ = diag(3,3,3)
    cov_corr = np.array(  # matrice piena, correlata
        [[1.5, 0.5, 0.2],
         [0.5, 1.5, 0.3],
         [0.2, 0.3, 1.5]]
    )

    shock_vars = [0.5, 1.0, 3.0]  # varianza dello shock

    df_grid = run_accuracy_grid(
        cov_x_list=[cov_diag1, cov_diag2, cov_corr],
        shock_var_list=shock_vars,
        n_runs=10,
        random_state=42,
        n_ind=3,
        n_env=3,
        n_samples=5000,
        noise_std=1,
        anomaly_rate=0.2,
        detector_kwargs={"contamination": 0.2, "bayesian_iter": 5}
    )

    print(df_grid)
