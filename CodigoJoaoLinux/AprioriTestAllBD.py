import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tabulate import tabulate
import os

df = pd.read_csv(r"TEA\archive\Autism_Screening_Data_Combined.csv")

df["Age"] = pd.cut(df["Age"], bins=[0, 3, 6, 100], labels=["0-3", "4-6", "7+"])

df["Sex"] = df["Sex"].map({"m": "Sexo: Masculino", "f": "Sexo: Feminino"})
df["Jauundice"] = df["Jauundice"].map({"yes": "Icterícia: Sim", "no": "Icterícia: Não"})
df["Family_ASD"] = df["Family_ASD"].map(
    {"yes": "Histórico familiar: Sim", "no": "Histórico familiar: Não"}
)
df["Class"] = df["Class"].map({"YES": "Autismo: Sim", "NO": "Autismo: Não"})

for i in range(1, 11):
    col = f"A{i}"
    df[col] = df[col].map({1: f"{col}: Sim", 0: f"{col}: Não"})

transacoes = []
for _, row in df.iterrows():
    itens = [row["Age"], row["Sex"], row["Jauundice"], row["Family_ASD"], row["Class"]]
    for i in range(1, 11):
        itens.append(row[f"A{i}"])
    transacoes.append(itens)

te = TransactionEncoder()
df_trans = te.fit(transacoes).transform(transacoes)
df_trans = pd.DataFrame(df_trans, columns=te.columns_)

items_frequentes = apriori(df_trans, min_support=0.2, use_colnames=True)
regras = association_rules(items_frequentes, metric="lift", min_threshold=1.5)

regras_autismo = regras[regras["consequents"].apply(lambda x: "Autismo: Sim" in x)]

colunas_saida = ["antecedents", "consequents", "support", "confidence", "lift"]
print(tabulate(regras_autismo[colunas_saida], headers="keys", tablefmt="pretty"))
