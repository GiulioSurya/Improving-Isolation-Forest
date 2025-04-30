import kagglehub

# Download latest version
path = kagglehub.dataset_download("camnugent/california-housing-prices")

print("Path to dataset files:", path)

import pandas as pd


# Leggi il file, specificando il separatore e i v2alori mancanti
df = pd.read_csv(
    "datas/housing.csv",
    sep=",",
    header=None  # Usa la prima riga (indice 0) come header
)




#save as csv
df.to_csv("datas/housing.csv", index=False)

# Leggi il file, specificando il separatore e i v2alori mancanti
df_new = pd.read_csv(
    "datas/housing.csv",
)
