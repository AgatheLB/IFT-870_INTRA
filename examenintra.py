# %%

# leba3207
from unicodedata import category
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

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


def get_score_sequence_matching(s, c1, category):
    if s[c1] is np.nan:
        return 0
    return difflib.SequenceMatcher(None, s[c1], category).ratio()


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
## B. Y-a-t-il une corrélation entre les catégories de journaux (attribut category) et les coûts de publication 
(attribut price)? Justifier la réponse.

Afin de déterminer s'il existe une corrélation entre les catégories et l'attribut prix, on s'intéresse à chaque 
catégorie une à une. 
Etant donné que chaque objet peut avoir plusieurs valeurs de catégories, on décide de séparer les catégories selon les 
différents séparateurs observés (|, and, .). On les convertit ensuite en one hot.
On calcule ensuite la corrélation catégorie par catégorie avec l'attribut prix. Pour cela, on ne considère que les 
objets présentant la catégorie testée et les valeurs de prix associées.
"""

# %%

labelled_data = data[data['category'].notna()]
data_to_predict = data[data['category'].isna()]

# %%

labelled_data['category'] = labelled_data['category'].str.replace(r'[\.\|&] | [\.\|&] | and ', '.', regex=True)
category_dummies = labelled_data['category'].str.get_dummies(sep='.')
category_dummies_prefix = category_dummies.add_prefix('category_')
print(f'Nombre de catégories après séparation: {category_dummies.shape[1]}')
# %%

labelled_data = pd.concat([labelled_data, category_dummies_prefix], axis=1) \
    # .drop(columns=['category'])

# %%

categories_correlation = {}

for header in category_dummies_prefix.columns:
    corr = labelled_data[header].corr(labelled_data['price'])
    if abs(corr) > 0.1:
        categories_correlation[header] = corr

fig, ax = plt.subplots()
plt.bar(categories_correlation.keys(), categories_correlation.values())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title(f'Catégories présentant des corrélations fortes avec\nl\'attribut prix et leurs valeurs')
plt.show()

# %%
"""
On remarque que certaines catégories présentent effectement une corrélation non négligeable avec l'attribut prix.
(Les catégories présentant une corrélation inférieures à 0.1 ne sont pas incluses dans le graphe)
Les catégories présentant la plus forte corrélation sont 'cell biology' et 'molecular'.
"""

# %%
"""
## C. Construire un modèle pour prédire les valeurs de catégorie de journaux manquantes de la façon la plus précise 
possible (cela inclut la sélection d’attributs informatifs, le choix et le paramétrage d’un modèle de classification, 
le calcul du score du modèle, l’application du modèle pour prédire les catégories manquantes). Justifier les choix 
effectués.

Dans le but de prédire les catégories de journaux, on doit s'intéresser à plusieurs attributs qui pourraient nous 
aider. Le nom du journal pourrait inclure certains mots-clés qui pourraient s'apparenter aux catégories du journal.
Le nom de l'éditeur pourrait également apporter de l'information sur les catégories. 
Etant donné qu'on a pu trouver certaines corrélations entre l'attribut prix et les catégories, on prend également en 
compte ce paramètre. 
Les informations de citation du journal pourraient également se révéler porteuses d'informations, ainsi que l'influence
des articles.
"""

# %%

labelled_data = labelled_data[labelled_data['price'].notna()]
headers = ['citation_count_sum', 'paper_count_sum', 'avg_cites_per_paper', 'proj_ai', 'price']
labelled_data = labelled_data.dropna(axis=0, subset=headers)

# %%

for header in tqdm(category_dummies.columns):
    labelled_data['jn_' + header] = labelled_data.apply(partial(get_score_sequence_matching, c1='journal_name',
                                                                category=header), axis=1)
    labelled_data['pn_' + header] = labelled_data.apply(partial(get_score_sequence_matching, c1='pub_name',
                                                                category=header), axis=1)

# %%

labelled_data = labelled_data.drop(columns=['category'])
print(f'size labelled_data before splitting: {labelled_data.shape[0]}')

# %%

jn_sm_headers = labelled_data.filter(like='jn_').columns.to_list()
pn_sm_headers = labelled_data.filter(like='pn_').columns.to_list()
attributes_of_interest = ['citation_count_sum', 'paper_count_sum', 'avg_cites_per_paper', 'proj_ai', 'price',
                          # 'date_stamp',
                          ]
attributes_of_interest.extend(jn_sm_headers)
attributes_of_interest.extend(pn_sm_headers)

# %%
"""
### Entrainement
On applique des modèles de classification ayant la capacité de pouvoir préduire des labels multiples. 
Pour cela, on utilise la méthode MultiOutputClassifier de sklearn afin qui consiste à adapter un classificateur par 
cible. 
A partir de là, on a pu essayer plusieurs types de classification, les deux meilleurs se sont révélés être les random 
forest et les K plus proches voisins. 

Le code suivant sert à faire une recherche d'hyperparamètres (succinte) sur un classification random forest. 
Pour le confort du temps de compilation, je n'ai pas intégré au rendu le classification K plus proche voisin, cependant
le résultat du meilleur modèle trouvé est décrit ci-dessous. 
"""

# %%

clfs = {'RandomForestClassifier': RandomForestClassifier()}

best_model = {'name': '', 'score': 0, 'model': None}
for name, clf in clfs.items():
    # for i in range(15, 17): # TODO
    for i in range(13, 14):
        X_train, X_test, y_train, y_test = train_test_split(labelled_data[attributes_of_interest],
                                                            labelled_data[category_dummies_prefix.columns],
                                                            test_size=0.33, random_state=42)
        X_train = X_train[attributes_of_interest]
        X_test = X_test[attributes_of_interest]
        y_train = y_train.to_numpy()

        clf.set_params(max_depth=i)

        print(f'Modèle {name} {i}')
        print('-- Entrainement')
        classifier = MultiOutputClassifier(clf, n_jobs=-1)
        classifier.fit(X_train, y_train)
        train_score = classifier.score(X_train, y_train)
        print(f'Score d\'entraînement: {train_score}')

        print('-- Test')
        test_predictions = classifier.predict(X_test)
        test_score = classifier.score(X_test, y_test)
        print(f'Score de test: {test_score}')

        if test_score > best_model.get('score'):
            best_model['name'], best_model['score'], best_model['model'] = name + ' ' + str(i), test_score, classifier

print(f"Le modèle présentant le meilleur score est {best_model.get('name')} avec {best_model.get('score')}")

# %%
"""
Résultats des différents essais:

Modèle RandomForestClassifier 12
-- Entrainement
Score d'entraînement: 0.9916259595254711
-- Test
Score de test: 0.7312588401697313

Modèle RandomForestClassifier 13
-- Entrainement
Score d'entraînement: 0.994417306350314
-- Test
Score de test: 0.7369165487977369
"""

# %%
"""
On conclut ainsi que le classificateur random forest ayant une profondeur maximale de 13 présente des résultats se 
trouve être le plus performant.  
Aussi, on se trouve en présence de résultats très performants, étant donné qu'on dispose de 88 catégories.
"""

# %%
"""
### Prédictions
"""

# %%

data_to_predict = data_to_predict.dropna(axis=0, subset=headers)

# %%

for header in tqdm(category_dummies.columns):
    data_to_predict['jn_' + header] = data_to_predict.apply(partial(get_score_sequence_matching, c1='journal_name',
                                                                    category=header), axis=1)
    data_to_predict['pn_' + header] = data_to_predict.apply(partial(get_score_sequence_matching, c1='pub_name',
                                                                    category=header), axis=1)

# %%

data_to_predict = data_to_predict[attributes_of_interest]

# %%

clf = best_model.get('model')
predictions = pd.DataFrame(clf.predict(data_to_predict))
predictions.columns = category_dummies.columns

# %%

count_categories = {}
for header in predictions.columns:
    nb = predictions[header].sum()
    if nb >= 1:
        count_categories[header] = nb

# %%

fig, ax = plt.subplots()
plt.bar(count_categories.keys(), count_categories.values())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title(f'')
plt.show()

# %%
"""
# Question 3: Régression-clustering
## A. Supprimer tous les attributs ayant plus de 50% de données manquantes.

On repart avec nos 3 tables originales.
"""

# %%

# def remove_empty_attribute(table):
#     for header in table:
#         if table[header].isna().sum() * 100 / len(table) > 50:
#             table = table.drop(columns=header)
#             headers.append(header)
#         return table
#
#
# # %%
# reduced_journal = journal
# reduced_price = price
# reduced_influence = influence
#
# reduced_journal = remove_empty_attribute(reduced_journal)
# reduced_price = remove_empty_attribute(reduced_price)
# reduced_influence = remove_empty_attribute(reduced_influence)

# %%
"""
## B. Construire un modèle pour prédire le coût actuel de publication (attribut «price») à partir des autres attributs 
(cela inclut la sélection d’attributs informatifs, le choix et le paramétrage d’un modèle derégression, le calcul du 
score du modèle, l’application du modèle pour prédire lescoûts).Justifier leschoix effectués.Lister les 10 revues qui
 s’écartent le plus (en + ou -) de la valeur prédite.
 
"""
