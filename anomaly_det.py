import pandas as pd
from sklearn.ensemble import IsolationForest
from  perturber import DatasetPreparer
from residual import ComputeResiduals
from utility import confusion_matrix, accuracy_score


class AnomalyDetector(object):

    def __init__(self,
                 env_cols: list[str],
                 ind_cols: list[str],
                 max_feature: float,
                 n_estimator: int,
                 random_state: int,
                 bayesian_iter: int = 10,
                 contamination: float = 0.1,
                 ):


        self.env_cols = env_cols
        self.ind_cols = ind_cols
        self.max_feature = max_feature
        self.n_estimator = n_estimator
        self.random_state = random_state
        self.bayesian_iter = bayesian_iter
        self.contamination = contamination

    def find_anomalies(self,
                       data: pd.DataFrame):

        self._fit_isolatio_forest()
        self.model.fit(data)

        data["anomaly"] = (self.model.predict(data) == -1).astype(int)

        return data

    def find_contextual_anomalies(self,
                                  data: pd.DataFrame):
        df = self._dataset_preparer(data=data)

        self._fit_isolatio_forest()
        self.model.fit(df)

        df['contextual_anomaly'] = (self.model.predict(df) == -1).astype(int)

        return df





# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

    def _residuals(self,
                   ind_col: str,
                   data: pd.DataFrame):
        """
        utilizza ResidualMaker per calcolare i residui per ogni variabile  di contesto
        """
        residual_maker = ComputeResiduals(
            env_cols=self.env_cols,
            ind_col= ind_col,
            n_bayes_iter=self.bayesian_iter,
            random_state= self.random_state
        )
        residuals = residual_maker.calc_residuals(data)

        return residuals

    def _dataset_preparer(self,
                          data: pd.DataFrame):
        """
        prende i residui e li unisce in un unico dataset
        :return:
        """
        #new dataset
        df_residual = pd.DataFrame()

        for ind in self.ind_cols:
            residuals = self._residuals(ind, data)
            print(f"residuals for {ind} calculated")
            df_residual[ind] = residuals

        return df_residual


    def _fit_isolatio_forest(self):
        """
        esegue il fit dell'isolation forest
        """
        model = IsolationForest(
            n_estimators=self.n_estimator,  # Numero di alberi nella foresta
            max_samples='auto',  # Numero di campioni da utilizzare per ogni albero
            contamination=self.contamination,  # Percentuale attesa di anomalie nel dataset
            max_features=self.max_feature,  # Percentuale di caratteristiche da utilizzare per ogni albero
            random_state=self.random_state,
        )

        self.model = model


if __name__ == "__main__":

    from ucimlrepo import fetch_ucirepo

    el_nino = fetch_ucirepo(id=122)
    X = el_nino.data.features
    datas = X.copy()
    datas.drop(columns=["date"], inplace=True)
    # remove na
    datas.dropna(inplace=True)

    env_col = ["year", "month", "day", "latitude", "longitude"]
    ind_col = ["zon_winds", "mer_winds", "humidity", "air_temp", "ss_temp"]

    prep = DatasetPreparer(
        env_cols=env_col,
        ind_cols=ind_col,
        random_state=42,
    )

    data_proc, split = prep.prepare(datas)

    anomaly = AnomalyDetector(
        env_cols=env_col,
        ind_cols=ind_col,
        max_feature=1,
        n_estimator=300,
        random_state = 42
    )

    #isolation forest
    df_iso_std = anomaly.find_anomalies(data_proc)

    conf_std_iso_anomaly = confusion_matrix(df_iso_std, data_proc, "anomaly", "is_anomaly")
    conf_std_iso_outlier = confusion_matrix(df_iso_std, data_proc, "anomaly", "is_outlier")

    df_iso_cad = anomaly.find_contextual_anomalies(data_proc)

    conf_cad_iso_anomaly = confusion_matrix(df_iso_cad, data_proc, "contextual_anomaly", "is_anomaly")
    conf_cad_iso_outlier = confusion_matrix(df_iso_cad, data_proc, "contextual_anomaly", "is_outlier")


