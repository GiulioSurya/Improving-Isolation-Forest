import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from perturber import DatasetPreparer
from skopt import BayesSearchCV
from skopt.space import Integer

class ComputeResiduals(object):
    def __init__(
            self,
            env_cols:list[str],
            ind_col: str,
            n_bayes_iter: int = 10,
            random_state: int = None):

        self.env_cols = list(env_cols)
        self.ind_col = ind_col
        self.n_bayes_iter = n_bayes_iter
        self.random_state = random_state

    def calc_residuals(self,
                       data: pd.DataFrame,):

        x = data[self.env_cols].values
        y = data[self.ind_col].values

        self._bayesian_search(data)

        residuals = np.zeros_like(y, dtype=float)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in kf.split(x):
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self._fit_random_forest(x_train, y_train)

            residuals[val_idx] = self._pred_residuals(x_val, y_val)

        return residuals



# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

    def _bayesian_search(self,
                         data: pd.DataFrame):

        y = data[self.ind_col]
        x = data[self.env_cols]

        base_model = RandomForestRegressor(random_state=self.random_state,
                                           n_jobs=-1)
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

        search_space = {
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(3, 30),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 10),
        }

        opt = BayesSearchCV(
            estimator=base_model,
            search_spaces=search_space,
            n_iter=self.n_bayes_iter,
            cv = kf,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1,
        )
        opt.fit(x, y)
        self.params = opt.best_params_

    def _fit_random_forest(self, x_train, y_train):
        model = RandomForestRegressor(
            **self.params,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model = model.fit(x_train, y_train)

    def _pred_residuals(self, x_pred, y_pred):

        y_hat = self.model.predict(x_pred)
        res = y_pred - y_hat
        return res



if __name__ == "__main__":
    from ucimlrepo import fetch_ucirepo

    el_nino = fetch_ucirepo(id=122)
    X = el_nino.data.features
    datas = X.copy()
    datas.drop(columns=["date"], inplace=True)
    datas.dropna(inplace=True)

    # 2. Preparazione dataset (flag + perturbazioni)
    prep = DatasetPreparer(
        env_cols=["year", "month", "day", "latitude", "longitude"],
        ind_cols=["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"],
        random_state=42,
    )
    prepared_df, _ = prep.prepare(datas)

    # 3. Calcolo dei residui sul target "zon_winds"
    env_cols = ["year", "month", "day", "latitude", "longitude"]
    res = ComputeResiduals(env_cols, "zon_winds", n_bayes_iter=1, random_state=42)
    residuals = res.calc_residuals(data=prepared_df)


    print("stop")



