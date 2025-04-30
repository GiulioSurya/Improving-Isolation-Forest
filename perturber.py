from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DatasetSplit:
    """
    classe container per i dataframe
    """

    indices: dict[str, pd.Index]

    def __getattr__(self, item: str):
        return self.indices[item]

    def as_df(self,
              data: pd.DataFrame,
              part: str):

        """Restituisce il sotto‐DataFrame relativo alla partizione richiesta."""
        return data.loc[self.indices[part]]


class DatasetPreparer:
    """Genera una singola suddivisione di dati usando **solo indici**.

    Il metodo :py:meth:`prepare` restituisce:

    ``prepared_data`` – una copia del dataset con le colonne flag (`is_anomaly`,
    `is_outlier`) **e le righe perturbate già modificate**;
    ``split`` – un :class:`DatasetSplit` con gli indici delle varie partizioni.
    """

    def __init__(self,
                 env_cols: list[str],
                 ind_cols: list[str],
                 test_size: float = 0.20,  # dal paper di Song
                 outlier_fraction: float = 0.20,  # dal paper di Song
                 random_state: int = None):

        self.env_cols = list(env_cols)
        self.ind_cols = list(ind_cols)
        self.test_size = test_size
        self.outlier_fraction = outlier_fraction
        self.random_state = random_state


    def prepare(self,
                data: pd.DataFrame):
        """Restituisce una **copia** del dataset con flag + split di indici."""


        df = data.copy()
        df = df.reset_index(drop=True)


        train_idx, test_idx = train_test_split(
            df.index,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # outlier detection
        test_env = df.loc[test_idx, self.env_cols]
        outlier_mask = self._identify_outliers_gmm(test_env)
        outliers_idx = test_env.index[outlier_mask]
        remaining_idx = test_env.index[~outlier_mask]

        # divide in perturbed and non-perturbed
        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(remaining_idx)
        half = len(shuffled) // 2
        perturbed_idx = pd.Index(shuffled[:half])
        nonperturbed_idx = pd.Index(shuffled[half:]).append(outliers_idx)

        # Apply perturbation
        self._apply_perturbation(df, perturbed_idx)

        # flags
        df["is_anomaly"] = 0
        df.loc[perturbed_idx, "is_anomaly"] = 1

        df["is_outlier"] = 0
        df.loc[outliers_idx, "is_outlier"] = 1

        split = DatasetSplit(
            {
                "train": pd.Index(train_idx),
                "perturbed": perturbed_idx,
                "nonperturbed": nonperturbed_idx,
                "outliers": outliers_idx,
            }
        )

        return df, split

    # ----------------------------------------------------------------------
    # Helper – outlier detection
    # ----------------------------------------------------------------------
    def _identify_outliers_gmm(self,
                               df_env: pd.DataFrame):

        numeric_df = df_env.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("Nessuna feature ambientale numerica per il GMM.")

        x_env = numeric_df.to_numpy()
        x_scaled = StandardScaler().fit_transform(x_env)

        max_k = min(5, len(x_scaled))
        best_bic = np.inf
        best_gmm = None

        for k in range(1, max_k + 1):
            gmm_k = GaussianMixture(
                n_components=k,
                covariance_type="full",
                reg_covar=1e-6,
                random_state=self.random_state,
                init_params="kmeans",
            ).fit(x_scaled)
            bic_k = gmm_k.bic(x_scaled)
            if bic_k < best_bic:
                best_bic = bic_k
                best_gmm = gmm_k

        gmm = best_gmm  # type: ignore[assignment]
        log_probs = gmm.score_samples(x_scaled)
        threshold = np.percentile(log_probs, 100 * self.outlier_fraction)
        return log_probs < threshold

    # ----------------------------------------------------------------------
    # Helper – perturbation (in‐place)
    # ----------------------------------------------------------------------
    def _apply_perturbation(self,
                            df: pd.DataFrame,
                            idx: pd.Index):

        k = min(50, max(1, len(idx) // 4))  # dal paper di song
        rng = np.random.default_rng(self.random_state)

        for i in idx:
            y_ind = df.loc[i, self.ind_cols]

            # candidati (indici diversi da i, all'interno di idx)
            candidates_idx = rng.choice(idx.drop(i), size=min(k, len(idx) - 1), replace=False)
            candidates = df.loc[candidates_idx, self.ind_cols].to_numpy()
            dists = np.linalg.norm(candidates - y_ind.to_numpy(), axis=1)
            y_prime = candidates[np.argmax(dists)]

            # aggiorna in‐place gli indicatori della riga i
            df.loc[i, self.ind_cols] = y_prime


if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo

    el_nino = fetch_ucirepo(id=122)
    datas = el_nino.data.features.copy()
    datas.drop(columns=["date"], inplace=True)
    datas.dropna(inplace=True)

    prep = DatasetPreparer(
        env_cols=["year", "month", "day", "latitude", "longitude"],
        ind_cols=["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"],
        random_state=42,
    )

    prepared_df, split = prep.prepare(datas)

