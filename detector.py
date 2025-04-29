
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import KFold



from old.perturber import DatasetPreparer, DatasetSplit  # type: ignore



from skopt import BayesSearchCV  # type: ignore
from skopt.space import Integer

class AnomalyDetector:
    """Rilevatore di anomalie RandomForest‑Residui + IsolationForest."""

    # ------------------------------------------------------------------
    # Costruttore
    # ------------------------------------------------------------------
    def __init__(
        self,
        data: pd.DataFrame,
        env_cols: List[str],
        ind_cols: List[str],
        *,
        test_size: float = 0.20,
        outlier_fraction: float = 0.20,
        random_state: Optional[int] = None,
        n_iter_bayes: int = 25,
        cv_folds: int = 3,
    ) -> None:
        self.data = data.copy()
        self.env_cols = list(env_cols)
        self.ind_cols = list(ind_cols)
        self.test_size = test_size
        self.outlier_fraction = outlier_fraction
        self.random_state = random_state
        self.n_iter_bayes = n_iter_bayes
        self.cv_folds = cv_folds

        # Inizializzazioni
        self.split: DatasetSplit | None = None
        self._dataset: pd.DataFrame | None = None
        self.rf_models: Dict[str, RandomForestRegressor] = {}
        self.residuals_: pd.DataFrame | None = None
        self.iso_forest_: IsolationForest | None = None
        self.scores_: pd.Series | None = None

    # ------------------------------------------------------------------
    # Pipeline esterna
    # ------------------------------------------------------------------
    def fit(self) -> "AnomalyDetector":
        """Esegue preprocessing, Random Forest, Isolation Forest."""
        self._preprocess()
        self._fit_random_forests()
        self._fit_isolation_forest()
        return self

    # ------------------------------------------------------------------
    # Accessor per punteggi e label
    # ------------------------------------------------------------------
    def anomaly_scores(self) -> pd.Series:
        if self.scores_ is None:
            raise RuntimeError("Chiama `.fit()` prima di accedere ai punteggi.")
        return self.scores_

    def labels(self, threshold: Optional[float] = None) -> pd.Series:
        scores = self.anomaly_scores()
        if threshold is None:
            # Usa le label interne (1 normale, -1 anomalo)
            return pd.Series(self.iso_forest_.predict(self.residuals_), index=scores.index)  # type: ignore
        return pd.Series(np.where(scores < threshold, -1, 1), index=scores.index)

    # ------------------------------------------------------------------
    # Predizione su nuovi dati
    # ------------------------------------------------------------------
    def predict(self, new_data: pd.DataFrame) -> pd.Series:
        """Calcola punteggi di anomalia per nuovi dati."""
        if self.iso_forest_ is None:
            raise RuntimeError("Il modello non è addestrato. Chiama `.fit()` prima.")

        missing_env = set(self.env_cols) - set(new_data.columns)
        missing_ind = set(self.ind_cols) - set(new_data.columns)
        if missing_env or missing_ind:
            raise ValueError(f"Colonne mancanti: {missing_env | missing_ind}")

        res_df = pd.DataFrame(index=new_data.index)
        X_new = new_data[self.env_cols]
        for y_col in self.ind_cols:
            y_true = new_data[y_col]
            y_pred = self.rf_models[y_col].predict(X_new)
            res_df[f"res_{y_col}"] = y_true - y_pred

        scores = -self.iso_forest_.score_samples(res_df)
        return pd.Series(scores, index=new_data.index)

    # ------------------------------------------------------------------
    # Step interni
    # ------------------------------------------------------------------
    def _preprocess(self) -> None:
        preparer = DatasetPreparer(
            data=self.data,
            env_cols=self.env_cols,
            ind_cols=self.ind_cols,
            test_size=self.test_size,
            outlier_fraction=self.outlier_fraction,
            random_state=self.random_state,
        )
        self.split = preparer.prepare()
        self._dataset = self.split.dataset()

    def _fit_random_forests(self) -> None:
        if self._dataset is None:
            raise RuntimeError("Preprocess non eseguito.")

        X_all = self._dataset[self.env_cols]
        self.residuals_ = pd.DataFrame(index=self._dataset.index)
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for y in self.ind_cols:
            y_all = self._dataset[y]
            base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            search_space = {
                "n_estimators": Integer(100, 500),
                "max_depth": Integer(3, 30),
                "min_samples_split": Integer(2, 10),
                "min_samples_leaf": Integer(1, 10),
            }
            opt = BayesSearchCV(
                estimator=base_model,
                search_spaces=search_space,
                n_iter=self.n_iter_bayes,
                cv=kf,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )
            opt.fit(X_all, y_all)
            best_rf = opt.best_estimator_
            self.rf_models[y] = best_rf
            y_pred = best_rf.predict(X_all)
            self.residuals_[f"res_{y}"] = y_all - y_pred

    def _fit_isolation_forest(self) -> None:
        if self.residuals_ is None:
            raise RuntimeError("Residui non calcolati.")

        self.iso_forest_ = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=self.random_state,
        )
        self.iso_forest_.fit(self.residuals_)
        self.scores_ = pd.Series(-self.iso_forest_.score_samples(self.residuals_), index=self.residuals_.index)


# -----------------------------------------------------------------------------
# Esempio uso
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo

    el_nino = fetch_ucirepo(id=122)

    # Copia del DataFrame per evitare SettingWithCopyWarning
    df = el_nino.data.features.copy()
    # Rimuoviamo la colonna "date" e le righe con NA in un'unica catena
    df = (
        df.drop(columns=["date"], errors="ignore")
          .dropna()
          .reset_index(drop=True)
    )

    env_cols = ["year", "month", "day", "latitude", "longitude"]
    ind_cols = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]

    detector = AnomalyDetector(
        data=df,
        env_cols=env_cols,
        ind_cols=ind_cols,
        random_state=42,
        n_iter_bayes=15,
    )
    detector.fit()

    print(detector.anomaly_scores().head())
