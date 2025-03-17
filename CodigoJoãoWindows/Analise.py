import pandas as pd

caminho =  r"TEA\archive\Toddler Autism dataset July 2018.csv"


ds_autismo_crianca = pd.read_csv(caminho)

ds_autismo_crianca = ds_autismo_crianca.drop("Case_No", axis=1)
ds_autismo_crianca = ds_autismo_crianca.drop("Age_Mons", axis=1)

pd.set_option('display.max_columns', None)

novos_nomes = {
    'A1': 'O seu filho olha para você quando você chama o nome dele/dela?',
    'A2': 'Quão fácil é para você obter contato visual com seu filho?',
    'A3': 'Seu filho aponta para indicar que quer algo? (por exemplo, um brinquedo fora de alcance)',
    'A4': 'Seu filho aponta para compartilhar um interesse com você? (por exemplo, apontar para uma cena interessante)',
    'A5': 'Seu filho faz de conta? (por exemplo, cuidar de bonecas, falar em um telefone de brinquedo)',
    'A6': 'Seu filho segue o que você está olhando?',
    'A7': 'Se você ou alguém da família estiver visivelmente chateado, seu filho demonstra sinais de querer confortar a pessoa? (por exemplo, acariciando o cabelo, abraçando)',
    'A8': 'Você descreveria as primeiras palavras do seu filho como:',
    'A9': 'Seu filho usa gestos simples? (por exemplo, acenar para se despedir)',
    'A10': 'Seu filho fica olhando para o nada sem um propósito aparente?'
}

ds_autismo_crianca = ds_autismo_crianca.rename(columns=novos_nomes)

for coluna in novos_nomes.values():
    ds_autismo_crianca[coluna] = ds_autismo_crianca[coluna].replace({1: 'Sim', 0: 'Não'})

#print(ds_autismo_crianca.columns)

ds_autismo_crianca['Class/ASD Traits '] = ds_autismo_crianca['Class/ASD Traits '].replace({'Yes': 'Sim', 'No': 'Não'})
ds_autismo_crianca['Jaundice'] = ds_autismo_crianca['Jaundice'].replace({'yes': 'Sim', 'no': 'Não'})
ds_autismo_crianca['Family_mem_with_ASD'] = ds_autismo_crianca['Family_mem_with_ASD'].replace({'yes': 'Sim', 'no': 'Não'})
# Substituir valores na coluna 'Ethnicity' com as traduções
ds_autismo_crianca['Ethnicity'] = ds_autismo_crianca['Ethnicity'].replace({
    'middle eastern': 'do Oriente Médio',
    'White European': 'Europeu Branco',
    'Hispanic': 'Hispânico',
    'black': 'Negro',
    'asian': 'Asiático',
    'south asian': 'Sul-Asiático',
    'Native Indian': 'Indígena Nativo',
    'Others': 'Outros',
    'Latino': 'Latino',
    'mixed': 'Misto',
    'Pacifica': 'Pacífico'
})
#print(ds_autismo_crianca['Ethnicity'].unique())
# Exibir as primeiras linhas para conferir
#print(ds_autismo_crianca[['Ethnicity']].head())

dados_restantes = ds_autismo_crianca.iloc[:, 11:]

#print(dados_restantes.head())
#print(dados_restantes.columns)

ds_autismo_crianca['Who completed the test'] = ds_autismo_crianca['Who completed the test'].replace({
    'family member': 'membro da família',
    'Health Care Professional': 'Profissional de Saúde',
    'Health care professional': 'Profissional de Saúde',
    'Self': 'Auto',
    'Others': 'Outros'
})

ds_autismo_crianca = ds_autismo_crianca.rename(columns={
    'Sex': 'Sexo',
    'Ethnicity': 'Etnia',
    'Jaundice': 'Icterícia',
    'Family_mem_with_ASD': 'Membro_família_TEA',
    'Who completed the test': 'Quem_Completou_Teste',
    'Class/ASD Traits ': 'Autismo'
})

#print(ds_autismo_crianca.head(5))

#print(ds_autismo_crianca.columns)


#           APLICANDO ALGORITMO APRIORI
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

ds_autismo_crianca_encoded = pd.get_dummies(ds_autismo_crianca, columns=[
    'O seu filho olha para você quando você chama o nome dele/dela?',
    'Quão fácil é para você obter contato visual com seu filho?',
    'Seu filho aponta para indicar que quer algo? (por exemplo, um brinquedo fora de alcance)',
    'Seu filho aponta para compartilhar um interesse com você? (por exemplo, apontar para uma cena interessante)',
    'Seu filho faz de conta? (por exemplo, cuidar de bonecas, falar em um telefone de brinquedo)',
    'Seu filho segue o que você está olhando?',
    'Se você ou alguém da família estiver visivelmente chateado, seu filho demonstra sinais de querer confortar a pessoa? (por exemplo, acariciando o cabelo, abraçando)',
    'Você descreveria as primeiras palavras do seu filho como:',
    'Seu filho usa gestos simples? (por exemplo, acenar para se despedir)',
    'Seu filho fica olhando para o nada sem um propósito aparente?',
    'Sexo',
    'Etnia',
    'Icterícia',
    'Membro_família_TEA',
    'Quem_Completou_Teste'
])

#frequent_itemsets = apriori(ds_autismo_crianca_encoded, min_support=0.1, use_colnames=True)

#association_rules_result = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

#print(association_rules_result)

print("Joao")
for coluna in ds_autismo_crianca_encoded.columns:
    valores_unicos = ds_autismo_crianca_encoded[coluna].unique()
    print(f"Coluna: {coluna} - Valores únicos: {valores_unicos}")