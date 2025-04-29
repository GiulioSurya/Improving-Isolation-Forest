import pandas as pd
from sklearn.ensemble import IsolationForest

from  perturber import DatasetPreparer
from residual import ComputeResiduals


class AnomalyDetector(object):

    def __init__(self,
                 env_cols: list[str],
                 ind_cols: list[str],
                 max_feature: float,
                 n_estimator: int,
                 ):


        self.env_cols = env_cols
        self.ind_cols = ind_cols
        self.max_feture = max_feature
        self.n_estimator = n_estimator

        pass

    def find_anomalies(self):
        """
        esegue un isolation forest senza utilizzare metodologia CAD
        :return:
        """
        pass

    def find_contextual_anomalies(self,
                                  data: pd.DataFrame):
        """
        esegue un isolation forest con metodologia CAD
        :return:
        """
        df = self._dataset_preparer(data=data)

        self._fit_isolatio_forest()
        self.model.fit(df)

        df['contextual_anomaly'] = self.model.predict(df)

        return df


#------------------------------------
#--------HELPER---------------------
#----------------------------------
    def _residuals(self,
                   ind_col: str,
                   data: pd.DataFrame):
        """
        utilizza ResidualMaker per calcolare i residui per ogni variabile  di contesto
        """
        residual_maker = ComputeResiduals(
            env_cols=self.env_cols,
            ind_col= ind_col
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
        :return:
        """
        model = IsolationForest(
            n_estimators=self.n_estimator,  # Numero di alberi nella foresta
            max_samples='auto',  # Numero di campioni da utilizzare per ogni albero
            contamination=0.1,  # Percentuale attesa di anomalie nel dataset
            max_features=self.max_feture,  # Percentuale di caratteristiche da utilizzare per ogni albero
            random_state=42
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
        n_estimator=300
    )

    df = anomaly.find_contextual_anomalies(data_proc)


    conf_matrix_ano = pd.crosstab(pd.Series(df["contextual_anomaly"]), pd.Series(data_proc["is_anomaly"]), rownames=["Predicted"], colnames=["Actual"])
    conf_matrix_out = pd.crosstab(pd.Series(df["contextual_anomaly"]), pd.Series(data_proc["is_outlier"]), rownames=["Predicted"], colnames=["Actual"])

