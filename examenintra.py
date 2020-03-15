# %%

# leba3207

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

standard_journal_categories = ["ANTHROPOLOGY", "ASTRONOMY AND ASTROPHYSICS", "CIRCUITS", "COMPUTER SCIENCE",
                               "DENTISTRY", "DERMATOLOGY", "ECOLOGY AND EVOLUTION", "ECONOMICS", "EDUCATION", "ENERGY",
                               "ENVIRONMENTAL HEALTH", "FOOD SCIENCE", "GASTROENTEROLOGY", "GEOTECHNOLOGY",
                               "HIGH ENERGY PHYSICS", "HISTORY AND PHILOSOPHY OF SCIENCE", "INFECTIOUS DISEASES",
                               "INFORMATION SCIENCE", "LAW", "LINGUISTICS", "MARKETING", "MATHEMATICS", "MEDICINE",
                               "MOLECULAR AND CELL BIOLOGY", "MYCOLOGY", "NEPHROLOGY", "NEUROSCIENCE", "ONCOLOGY",
                               "OPERATIONS RESEARCH", "OPHTHALMOLOGY", "ORTHOPEDICS", "PHARMACOLOGY",
                               "PHYSICS AND CHEMISTRY", "PLANT BIOLOGY", "PLASTIC SURGERY", "POLITICAL SCIENCE",
                               "PSYCHIATRY", "PSYCHOLOGY", "RADIOLOGY", "RHEUMATOLOGY", "ROBOTICS", "SOCIOLOGY",
                               "SPORTS MEDICINE", "STRUCTURAL ENGINEERING", "UROLOGY", "VETERINARY", "WOOD PRODUCTS"]

# %%
"""
Chargement des données des différents fichiers
"""

# %%

data_folder = 'data/'

journal_file = 'api_journal11-13-17.csv'
price_file = 'api_price11-13-17.csv'
influence_file = 'estimated-article-influence-scores-2015.csv'

journal = pd.read_csv(data_folder + journal_file, sep=',', encoding='latin1')
price = pd.read_csv(data_folder + price_file, sep=',', index_col=0)
influence = pd.read_csv(data_folder + influence_file, sep=',', index_col=0)


# %%

def get_uniqueness_attributes(table):
    return table.nunique(axis=0)


def get_ratio_missing_values(table):
    return table.isnull().sum() * 100 / len(table)


def get_unique_values_of_attribute(table, header):
    return table[header].unique()


def lowercase_columns(table, headers):
    for header in headers:
        table[header] = table[header].str.lower()


# TODO: include headers specification
def get_df_duplicated_rows_dropped(table):
    return table.drop_duplicates()


def plot_categories_frequency(table, header):
    fig, ax = plt.subplots()
    table[header].value_counts()[0:5].plot(ax=ax, kind='bar')
    plt.title(f'Fréquence d\'apparition des catégories de l\'attribut {header}')
    plt.show()


def get_mean_price_per_year():
    mean_price_per_year = {}
    for index, p in price.iterrows():
        year = p['date_stamp'].year
        if year in mean_price_per_year:
            mean_price_per_year[year] += p['price']
        else:
            mean_price_per_year[year] = p['price']

    for year, value in mean_price_per_year.items():
        mean_price_per_year[year] /= len(price[price['price'] != 0])
    return {k: v for k, v in sorted(mean_price_per_year.items(), key=lambda item: item[1], reverse=True)}


def rename_df_headers(table, dict_headers):
    return table.rename(columns=dict_headers)


# %%

"""
# Question 1: Exploration-Description
## Présenter une description de chacun des attributs des 3 tables, avec des graphiques pour la visualisation des 
statistiques descriptives au besoin.

### Table journal
"""

# %%
# TODO: visualisations
# fréquence des valeurs

# %%

print(journal.head())

# %%

"""
issn: identifiant du journal
Les valeurs de cet attribut semblent suivre un format particulier tel que: 4 digits - 4 digits

journal_name: nom textuel du journal
Les valeurs sont textuelles, ne suivant pas de valeurs catégorielles particulière à priori.

pub_name: nom de l'éditeur du journal
Les valeurs sont textuelles, ne suivant pas de valeurs catégorielles particulière à priori.

is_hybrid: indique si le journal est hybride, ce qui signifie que c'est une revue sur abonnement dont certains articles
sont en libre accès, comme l'indique le site http://flourishoa.org/about.

category: renseigne sur la/les catégorie(s) des sujets abordés par le journal 
Les valeurs sont textuelles et sont catégorielles. Chaque objet peut posséder des valeurs multiples pour cet attribut. La séparation entre les différentes valeurs semblent
être inconsistante.

url: indique l'adresse url du site du journal


Les attributs journal_name, pub_name et category étant des données textuelles très inconsistantes, je décide avant tout
traitement et étude supplémentaire de transformer les valeurs en minuscule pour limiter au maximum l'inconsistence.
"""

# %%

lowercase_columns(journal, ['journal_name', 'pub_name', 'category'])

# %%

print(f"Valeurs uniques des attributs de journal:\n"
      f"{get_uniqueness_attributes(journal)}")
print(f"Ratio de valeurs vides pour les attributs de journal:\n"
      f"{get_ratio_missing_values(journal)}")

print(f"Valeurs possibles pour l'attribut is_hybrid de journal:\n"
      f"{get_unique_values_of_attribute(journal, 'is_hybrid')}")
print(f"Valeurs possibles pour l'attribut category de journal:\n"
      f"{get_unique_values_of_attribute(journal, 'category')}")

# %%
"""
Les attributs category et url présentent un nombre conséquent de valeurs manquantes.
L'attribut issn présente des valeurs uniques pour chacun de ses objets.

On remarque qu'il existe uniquement deux valeurs pour l'attribut is_hybrid (soit 1 soit 0). 
"""

# %%

print(f"Valeurs possibles de pub_name quand is_hybrid vaut 1:\n"
      f"{journal[journal['is_hybrid'] == 1]['pub_name'].unique()}")

# %%

plot_categories_frequency(journal, 'pub_name')
plot_categories_frequency(journal, 'category')

# %%
"""
### Table price
"""

# %%

print(price.head())

# %%
"""

price: information du prix d'une publication pour le journal associé à une date précise, en dollar US
Si celui-ci est à 0, on peut coompendre que celui-ci est gratuit

date_stamp: horodatage de l'information de prix d'une publication, en format années-mois-jour

journal_id: identifiant du journal
Les valeurs semblent suivre consistantement un format du type: 4 digits - 4 digits

influence_id: identifiant de l'influence
Les valeurs suivent un format 4 digits.

url: indique l'adresse url du site de l'auteur

license: indique le type de license utilisé par le journal pour les différents articles utilisés.


On convertit l'attribut date_stamp en type date.
"""

# %%

price['date_stamp'] = pd.to_datetime(price['date_stamp'], errors='coerce', format='%Y-%m-%d')

# %%

print(f"Valeurs uniques des attributs de price:\n"
      f"{get_uniqueness_attributes(price)}")
print(f"Ratio de valeurs vides pour les attributs de price:\n"
      f"{get_ratio_missing_values(price)}")

print(f"Exemples de valeurs possibles pour l'attribut influence_id de price:\n"
      f"{get_unique_values_of_attribute(price, 'influence_id')[1:8]}")
print(f"Valeurs possibles pour l'attribut license de price:\n"
      f"{get_unique_values_of_attribute(price, 'license')}")

# %%
"""
Les attributs influence_id, url et license présentent une majorité de valeurs manquantes.
"""

# %%

mean_price_per_year = get_mean_price_per_year()

plt.bar(range(len(mean_price_per_year)), mean_price_per_year.values())
plt.xticks(range(len(mean_price_per_year)), mean_price_per_year.keys())
plt.title("Moyenne par année des prix des publications")
plt.show()

# %%
"""
### Table influence
"""

# %%

print(influence.head())

# TODO: proj_ai, proj_ai_year
# %%
"""

journal_name: nom textuel du journal
Les valeurs sont textuelles, ne suivant pas de valeurs catégorielles particulière à priori.

issn: identifiant du journal
Les valeurs de cet attribut semblent suivre un format particulier tel que: 4 digits - 4 digits

citation_count_sum: indique le nombre de citations du jounal

paper_count_sum: indique le nombre de citations des articles du jounal

avg_cites_per_paper: indique la moyenne des citations par papier qui sont contenus du journal

proj_ai:

proj_ai_year:

"""

# %%

print(f"Valeurs uniques des attributs de influence:\n"
      f"{get_uniqueness_attributes(influence)}")
print(f"Ratio de valeurs vides pour les attributs de influence:\n"
      f"{get_ratio_missing_values(influence)}")

# %%

"""
# Question 2: Prétraitement-Représentation
## A. Effectuer un prétraitement des données pour supprimer les duplications et corriger les incohérences s’il y en a.
"""

# %%
"""
### Table journal
Dans un premier temps, on élimine les objets présentant des objets dupliqués sur tous les attributs.
On se base sur l'attribut issn qui devrait être unique pour chaque objet de la table, on vérifie son unicité.
"""

# %%

nb = len(journal)
journal = get_df_duplicated_rows_dropped(journal)
print(f"Nombre d'objets dupliqués éliminés dans journal: {nb - len(journal)}")

check = np.logical_not(journal['issn'].duplicated().any())
print(f"Unicité de l'attribut issn dans la table journal: {check}")

# %%
"""
### Table price
Etant donné que les index étaient fournis dans le fichier original et qu'on les utilise afin d'indexer nos objets, 
on vérifie qu'il n'existe pas de duplicata.

Ensuite, on élimine les objets présentant des objets dupliqués sur tous les attributs.
Dans un second temps, dans la table price, les objets se doivent d'être uniques selon deux attributs, date_stamp et
journal_id. S'ils ne le sont pas, alors ceux sont des objets dupliqués.
"""

# %%

check = np.logical_not(price.index.duplicated().any())
print(f"Unicité des indexes de la table price: {check}")

nb = len(price)
price = get_df_duplicated_rows_dropped(price)
print(f"Nombre d'objets dupliqués éliminés dans price: {nb - len(price)}")

duplicated_rows = price[price[['date_stamp', 'journal_id']].duplicated(keep=False)]

# %%
"""
Il existe des duplicata ambigus que l'on décide de traiter un à un.

Premier cas: un des objets présente un prix nul, on décide de choisir de l'éliminer au profit de l'autre.
"""

# %%

print(duplicated_rows.iloc[0].fillna(0) == duplicated_rows.iloc[1].fillna(0))
price = price.drop(duplicated_rows.index.values[0])

# %%
"""
Deuxième cas: leur valeur du prix est différente d'un léger écart, on décide de garder la deuxième de manière
arbitraire
"""

# %%

print(duplicated_rows.iloc[40].fillna(0) == duplicated_rows.iloc[41].fillna(0))
price = price.drop(duplicated_rows.index.values[40])

# %%
"""
Troisième cas: seule la valeur de license est différente, on décide de garder la première de manière arbitraire.
"""

# %%

for i in range(3, 39, 2):
    price = price.drop(duplicated_rows.index.values[i])

# %%
"""
### Table influence
Etant donné que les index étaient fournis dans le fichier original et qu'on les utilise afin d'indexer nos objets, 
on vérifie qu'il n'existe pas de duplicata.

On se base sur l'attribut issn qui devrait être unique pour chaque objet de la table, on vérifie son unicité.
"""

# %%

nb = len(influence)
influence = get_df_duplicated_rows_dropped(influence)
print(f"Nombre d'objets dupliqués éliminés dans influence: {nb - len(influence)}")

check = np.logical_not(influence['issn'].duplicated().any())
print(f"Unicité de l'attribut issn dans la table influence: {check}")

# %%
"""
## Merge
Afin de simplifier les opérations, on génère une seule table reprenant les informations des trois tables.
On vérifie d'abord si les identifiants communs aux différentes tables sont présentes dans les tables à merger.
En premier, on vérifie si les valeurs de l'attribut issn de influence sont existantes dans l'attribut du même nom 
dans journal.
De même, on vérifie les valeurs de l'attribut journal_id de price sont existantes dans l'attribut issn dans journal. 
"""
# %%

check = influence['issn'].isin(journal['issn']).any()
print(f"Pas de valeur d'issn manquante dans journal par rapport à influence : {check}")

check = price['journal_id'].isin(journal['issn']).any()
print(f"Pas de valeur d'issn manquante dans journal par rapport à price : {check}")

# %%
"""
On applique maintenant le merge des trois tables en deux étapes. D'abord, on merge influence dans journal, puis price
est ensuite mergé dans le résultat du premier merge.
"""
# %%

price = rename_df_headers(price, {"journal_id": "issn", "url": "url_autor"})
journal = rename_df_headers(journal, {"url": "url_journal"})

temp = pd.merge(journal, influence, on='issn')
check = len(temp[temp['journal_name_x'] != temp['journal_name_y']])
print(f"Nombre d'aberrances entre les valeurs journal_name des tables journal et influence: {check}")
temp = temp.drop(columns=['journal_name_y'])
temp = rename_df_headers(temp, {"journal_name_x": "journal_name"})

data = pd.merge(price, temp, on='issn')
data = get_df_duplicated_rows_dropped(data)

print(f"Valeurs uniques des attributs de data:\n"
      f"{get_uniqueness_attributes(data)}")
print(f"Ratio de valeurs vides pour les attributs de data:\n"
      f"{get_ratio_missing_values(data)}")

# %%
"""
## B. Y-a-t-il une corrélation entre les catégories de journaux (attribut category) et les coûts de publication 
(attribut price)? Justifier la réponse.
"""
