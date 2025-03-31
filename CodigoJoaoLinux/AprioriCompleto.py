import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from tabulate import tabulate

df = pd.read_csv("TEA/archive/Autism_Screening_Data_Combined.csv")
novos_nomes = {
    "A1": "O seu filho olha para você quando você chama o nome dele/dela?",
    "A2": "Quão fácil é para você obter contato visual com seu filho?",
    "A3": "Seu filho aponta para indicar que quer algo? (por exemplo, um brinquedo fora de alcance)",
    "A4": "Seu filho aponta para compartilhar um interesse com você? (por exemplo, apontar para uma cena interessante)",
    "A5": "Seu filho faz de conta? (por exemplo, cuidar de bonecas, falar em um telefone de brinquedo)",
    "A6": "Seu filho segue o que você está olhando?",
    "A7": "Se você ou alguém da família estiver visivelmente chateado, seu filho demonstra sinais de querer confortar a pessoa? (por exemplo, acariciando o cabelo, abraçando)",
    "A8": "Você descreveria as primeiras palavras do seu filho como:",
    "A9": "Seu filho usa gestos simples? (por exemplo, acenar para se despedir)",
    "A10": "Seu filho fica olhando para o nada sem um propósito aparente?",
}

df = df.rename(columns=novos_nomes)
df["Age"] = pd.cut(
    df["Age"],
    bins=[0, 12, 17, 59, 100],
    labels=["Criança", "Adolescente", "Adulto", "Idoso"],
)
df["Sex"] = df["Sex"].map({"m": "Sexo: Masculino", "f": "Sexo: Feminino"})
df["Jauundice"] = df["Jauundice"].map({"yes": "Icterícia: Sim", "no": "Icterícia: Não"})
df["Family_ASD"] = df["Family_ASD"].map(
    {"yes": "Histórico familiar TEA: Sim", "no": "Histórico familiar TEA: Não"}
)
df["Class"] = df["Class"].map({"YES": "Autismo: Sim", "NO": "Autismo: Não"})

df_encoded = pd.get_dummies(
    df, columns=["Age", "Sex", "Jauundice", "Family_ASD", "Class"], dtype=bool
)

items_frequentes = apriori(df_encoded, min_support=0.2, use_colnames=True)
regras = association_rules(items_frequentes, metric="lift", min_threshold=1.0)

regras_autismo = regras[
    regras["consequents"].apply(lambda x: "Class_Autismo: Sim" in x)
]

colunas_saida = ["antecedents", "consequents", "support", "confidence", "lift"]
regras_autismo_sorted = regras_autismo.sort_values(by="confidence", ascending=False)

if not regras_autismo.empty:

    print(tabulate(regras_autismo_sorted[colunas_saida].head(12), headers="keys", tablefmt="pretty"))
else:
    print("Nenhuma regra encontrada com consequente 'Autismo: Sim'.")