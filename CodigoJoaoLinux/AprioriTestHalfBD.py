import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tabulate import tabulate

caminho_base_dados = r"TEA\archive\Toddler Autism dataset July 2018.csv"
df = pd.read_csv(caminho_base_dados)

print("Metade",df.columns)
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

df = df.rename(
    columns={
        "Sex": "Sexo",
        "Ethnicity": "Etnia",
        "Jaundice": "Icterícia",
        "Family_mem_with_ASD": "Membro_família_TEA",
        "Who completed the test": "Quem_Completou_Teste",
        "Class/ASD Traits ": "Autismo",
    }
)


colunas_para_remover = ["Case_No"]
df.drop(columns=colunas_para_remover, errors="ignore", inplace=True)

df_encoded = pd.get_dummies(
    df,
    columns=[
        "Etnia",
        "Icterícia",
        "Membro_família_TEA",
        "Quem_Completou_Teste",
        "Autismo",
        "Sexo",
    ],
)

df_encoded["Age_Mons"] = pd.cut(
    df_encoded["Age_Mons"],
    bins=[0, 24, 36, 48],
    labels=["0-2 anos", "3 anos", "4 anos"],
)
df_encoded["Qchat-10-Score"] = pd.cut(
    df_encoded["Qchat-10-Score"], bins=[0, 5, 10], labels=["Baixa", "Alta"]
)

df_encoded = pd.get_dummies(df_encoded, columns=["Age_Mons", "Qchat-10-Score"])

df_encoded = df_encoded.loc[:, (df_encoded.sum() > 5)]

df_transacoes = df_encoded.astype(bool)

freq_items = apriori(df_transacoes, min_support=0.6, use_colnames=True)
regras = association_rules(freq_items, metric="lift", min_threshold=0.8)

regras["antecedents"] = regras["antecedents"].apply(lambda x: ", ".join(list(x)))
regras["consequents"] = regras["consequents"].apply(lambda x: ", ".join(list(x)))

colunas = ["antecedents", "consequents", "support", "confidence", "lift"]

print(tabulate(regras[colunas], headers="keys", tablefmt="pretty"))
