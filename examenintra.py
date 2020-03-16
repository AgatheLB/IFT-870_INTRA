# %%

# leba3207

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

import difflib
from functools import partial

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

# TODO: proj_ai moyenne
# %%
"""

journal_name: nom textuel du journal
Les valeurs sont textuelles, ne suivant pas de valeurs catégorielles particulière à priori.

issn: identifiant du journal
Les valeurs de cet attribut semblent suivre un format particulier tel que: 4 digits - 4 digits

citation_count_sum: indique le nombre de citations du jounal

paper_count_sum: indique le nombre de citations des articles du jounal

avg_cites_per_paper: indique la moyenne des citations par papier qui sont contenus du journal

proj_ai: information sur le score d'influence des articles du journal

proj_ai_year: spécification de l'année où l'information sur le score d'influence des articles du journal a été établie

"""

# %%

print(f"Valeurs uniques des attributs de influence:\n"
      f"{get_uniqueness_attributes(influence)}")
print(f"Ratio de valeurs vides pour les attributs de influence:\n"
      f"{get_ratio_missing_values(influence)}")

print(f"Valeurs possibles pour l'attribut proj_ai_year de influence:\n"
      f"{get_unique_values_of_attribute(influence, 'proj_ai_year')}")

# %%
"""
L'attribut proj_ai_year ne présentant qu'une seule valeur nous indique que les valeurs de l'attribut proj_ai ont toutes
été établies à la même période. 
"""
# TODO: proj_ai sum in function of journal
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
Les attributs citation_count_sum, paper_count_sum, avg_cites_per_paper et proj_ai présentent des valeurs nulles que l'on
décide d'attribuer à 0.
"""

# %%

headers = ['citation_count_sum', 'paper_count_sum', 'avg_cites_per_paper', 'proj_ai']
for header in headers:
    influence[header] = influence[header].fillna(0)

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

temp = pd.merge(journal, influence, on='issn', how='outer')
check = len(temp[temp['journal_name_x'] != temp['journal_name_y']])
print(f"Nombre d'aberrances entre les valeurs journal_name des tables journal et influence: {check}")
temp = temp.drop(columns=['journal_name_y'])
temp = rename_df_headers(temp, {"journal_name_x": "journal_name"})

print(f"Valeurs uniques des attributs de temp:\n"
      f"{get_uniqueness_attributes(temp)}")

data = pd.merge(temp, price, on=['issn'], how='outer')
data = get_df_duplicated_rows_dropped(data)

print(f"Valeurs uniques des attributs de data:\n"
      f"{get_uniqueness_attributes(data)}")
print(f"Ratio de valeurs vides pour les attributs de data:\n"
      f"{get_ratio_missing_values(data)}")

# %%
"""
On s'assure bien que les valeurs de l'attribut issn de la nouvelle date (data) sont uniques.
"""

# %%
"""
## B. Y-a-t-il une corrélation entre les catégories de journaux (attribut category) et les coûts de publication (attribut price)? Justifier la réponse.
"""

# %%

labelled_data = data[data['category'].notna()]
data_to_predict = data[data['category'].isna()]

# %%

labelled_data['category'] = labelled_data['category'].str.replace(r'[\.\|&] | [\.\|&] ', '.', regex=True)
category_dummies = labelled_data['category'].str.get_dummies(sep='.')
category_dummies = category_dummies.add_prefix('category_')

# %%

labelled_data = pd.concat([labelled_data, category_dummies], axis=1) \
    # .drop(columns=['category'])

# %%

categories_correlation = {}

for header in category_dummies.columns:
    corr = labelled_data[header].corr(labelled_data['price'])
    if abs(corr) > 0.1:
        categories_correlation[header] = corr

fig, ax = plt.subplots()
plt.bar(categories_correlation.keys(), categories_correlation.values())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title(f'Catégories présentant des corrélations fortes avec l\'attribut prix et leurs valeurs')
plt.show()

# %%
"""
## C. Construire un modèle pour prédire les valeurs de catégorie de journaux manquantes de la façon la plus précise 
## possible (cela inclut la sélection d’attributs informatifs, le choix et le paramétrage d’un modèle de classification, 
## le calcul du score du modèle, l’application du modèle pour prédire les catégories manquantes). Justifier les choix 
## effectués.

"""


# %%
# TODO: get key words from journal_name, publisher

def get_score_sequence_matching(s, c1, c2):
    return difflib.SequenceMatcher(None, s[c1], s[c2]).ratio()


# %%

labelled_data = labelled_data[labelled_data['price'].notna()]

headers = ['citation_count_sum', 'paper_count_sum', 'avg_cites_per_paper', 'proj_ai', 'price']
for header in headers:
    labelled_data[header] = labelled_data[header].fillna(0)

# %%

journal_name_score_match_category = pd.Series()
publisher_name_score_match_category = pd.Series()

labelled_data['journal_name_sm_category'] = labelled_data.apply(partial(get_score_sequence_matching, c1='journal_name',
                                                                        c2='category'), axis=1)
labelled_data['pub_name_sm_category'] = labelled_data.apply(partial(get_score_sequence_matching, c1='pub_name',
                                                                    c2='category'), axis=1)
# %%

labelled_data = labelled_data.drop(columns=['category'])
print(f'size labelled_data before splitting: {labelled_data.shape[0]}')
# %%

# vectorized_category_dummies = pd.DataFrame(columns=['category'])
# for index, value in labelled_data[category_dummies.columns].iterrows():
#     vectorized_category_dummies.at[index, 'category'] = value.ravel().tolist()
#
# vectorized_labelled_data = labelled_data
# vectorized_labelled_data['category'] = vectorized_category_dummies['category']

# %%
attributes_of_interest = ['citation_count_sum', 'paper_count_sum', 'avg_cites_per_paper', 'proj_ai', 'price',
                          # 'date_stamp',
                          'journal_name_sm_category', 'pub_name_sm_category']
headers = category_dummies.columns.tolist()
# headers.append('category')

X_train, X_test, y_train, y_test = train_test_split(labelled_data
                                                    .drop(columns=headers),
                                                    # labelled_data['category'],
                                                    labelled_data[category_dummies.columns],
                                                    test_size=0.33, random_state=42)

X_train = X_train[attributes_of_interest]
X_test = X_test[attributes_of_interest]


# %%
"""
"""
# TODO: justification


# %%

# mlb = MultiLabelBinarizer()
# mlb.fit_transform(y_train)

# %%

y_train = y_train.to_numpy()

# %%


# classifier = OneVsRestClassifier(clf)

# %%

clfs = {'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
        'RandomForestClassifier':  RandomForestClassifier(max_depth=10, random_state=1)}

best_model = {'name': '', 'score': 0, 'model': None}

for name, clf in clfs.items():
    print(f'Modèle {name}')
    print('## Entrainement ##')
    classifier = MultiOutputClassifier(clf, n_jobs=-1)
    classifier.fit(X_train, y_train)
    train_score = classifier.score(X_train, y_train)
    print(f'Score d\'entraînement: {train_score}')

    print('## Test ##')
    test_predictions = classifier.predict(X_test)
    test_score = classifier.score(X_test, y_test)
    print(f'Score de test: {test_score}')

    if test_score > best_model.get('score'):
        best_model['name'], best_model['score'], best_model['model'] = name, test_score, clf

print(f"Le modèle présentant le meilleur score est {best_model.get('name')} avec {best_model.get('score')}")

# %%
"""
Résultats des différents essais:

-> Essais avec price nan fixés à 0:
    MultiOutputClassifier + RandomForestClassifier = ~0%

-> Essais avec price nan droped:
    MultiOutputClassifier + RandomForestClassifier(10) = 50% 40% 
    MultiOutputClassifier + KNeighborsClassifier(2) = 40% 22%
    MultiOutputClassifier + KNeighborsClassifier(3) = 50% 27%
    MultiOutputClassifier + KNeighborsClassifier(5) = 30% 20%
"""

# %%
# TODO: REVELATIONS
"""
méthode 1:
calculer score sequence matcher avec seulement les catégories splittées et insérer chaque score dans la table data

méthode 2:
calcul score sequence matcher avec seulement les catégories splittées et garder information de la catégorie ayant le
meilleur score
"""

# %%
"""
# Question 3: Régréssion-clustering
## A. Supprimer tous les attributs ayant plus de 50% de données manquantes.
"""

# %%


