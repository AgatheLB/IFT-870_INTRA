# %%

# leba3207

from tqdm import tqdm
import heapq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import difflib
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

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

journal.name = 'journal'
price.name = 'price'
influence.name = 'influence'


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


def get_df_duplicated_rows_dropped(table):
    return table.drop_duplicates()


def plot_categories_frequency(table, header):
    fig, ax = plt.subplots()
    table[header].value_counts()[0:5].plot(ax=ax, kind='bar')
    plt.title(f'Présentation des 5 valeurs les plus fréquentes\nde l\'attribut {header} pour la table {table.name}')
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
    return {k: v for k, v in sorted(mean_price_per_year.items(), key=lambda item: item[0], reverse=True)}


def rename_df_headers(table, dict_headers):
    return table.rename(columns=dict_headers)


def get_score_sequence_matching(s, c1, category):
    if s[c1] is np.nan:
        return 0
    return difflib.SequenceMatcher(None, s[c1], category).ratio()


def get_empty_attribute_to_remove(table):
    headers = []
    for header in table.columns:
        if table[header].isna().sum() * 100 / len(table) > 50:
            headers.append(header)
    return headers


# %%
"""
# Question 1: Exploration-Description
## Présenter une description de chacun des attributs des 3 tables, avec des graphiques pour la visualisation des statistiques descriptives au besoin.

### Table journal
"""

# %%

print(journal.head())

# %%

"""
issn: identifiant de la revue
Les valeurs de cet attribut semblent suivre un format particulier tel que: 4 digits - 4 digits
Etant donné que l'identifiant spécifie chaque revue, cet attribut devrait présenter des valeurs uniques pour chacun
des objets.

journal_name: nom textuel de la revue
Les valeurs sont textuelles, ne suivant pas de valeurs catégorielles particulière à priori. Il n'existe pas de format
spécifié, les valeurs s'en retrouvent donc très inconsistantes.

pub_name: nom de l'éditeur de la revue
Les valeurs sont textuelles, ne suivant pas de valeurs catégorielles particulière à priori. Il n'existe pas de format
spécifié, les valeurs s'en retrouvent donc très inconsistantes.

is_hybrid: indique si la revue est hybride. Si oui, cela signifie que cette revue est disponible par abonnement où 
certains articles sont en libre accès et d'autres payants, comme l'indique le site http://flourishoa.org/about.

category: renseigne sur la/les catégorie(s) des sujets abordés par la revue
Les valeurs sont textuelles et sont catégorielles. Chaque objet peut posséder des valeurs multiples pour cet attribut. 
La séparation entre les différentes valeurs semblent être inconsistante.

url: indique l'adresse web url du site de la revue


Les attributs journal_name, pub_name et category étant des données textuelles très inconsistantes, je décide avant tout
traitement et étude supplémentaire de transformer les valeurs en minuscule pour limiter au maximum l'inconsistence
inutile entre les différentes valeurs.
"""

# %%

lowercase_columns(journal, ['journal_name', 'pub_name', 'category'])

# %%

print(f"Valeurs uniques des attributs de journal présentant {journal.shape[0]} objets:\n"
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
L'attribut issn présente, comme souhaité,  des valeurs uniques pour chacun de ses objets.

On remarque qu'il existe uniquement deux valeurs pour l'attribut is_hybrid (soit 1 soit 0), ce qui semble logique. 
"""

# %%

print(f"Valeurs possibles de pub_name quand is_hybrid vaut 1:\n"
      f"{journal[journal['is_hybrid'] == 1]['pub_name'].unique()}")

# %%
"""
Cela nous montre que parmi le grand nombre d'éditeurs possibles, seuls 4 permettent des revues hybrides.
"""

# %%

plot_categories_frequency(journal, 'pub_name')
plot_categories_frequency(journal, 'category')

# %%
"""
Malgré l'inconsistence des valeurs de ces deux attributs, on s'aperçoit néanmoins que certaines catégories et éditeurs
sont plus fréquents que d'autres.

Cette table mériterait un travail sur l'inconsistence des valeurs de category afin de pouvoir approfondir l'étude de cet
attribut.
"""

# %%
"""
### Table price
"""

# %%

print(price.head())

# %%
"""
price: information du coût d'une publication pour la revue, en dollar US, à une date précisée dans l'attribut date_stamp
Si celui-ci est à 0, on peut compendre qu'une publication au sein de cette revue est gratuite.

date_stamp: horodatage de l'information de coût d'une publication au sein d'une revue donnée.
Cet attribut suit un format années-mois-jour

journal_id: identifiant de la revue (permet la liaison avec la table journal, qui présente le même attribut sous le nom
issn)
Les valeurs semblent suivre consistantement un format du type: 4 digits - 4 digits. 

influence_id: identifiant de l'influence (permet la liaison avec la table influence afin d'avoir des informations sur 
l'influence d'une revue)
Les valeurs suivent un format 4 digits.

url: indique l'adresse web url du site de l'auteur

license: indique le type de license utilisé par la revue pour les différents articles qui y sont publiés.


Comme précisé dans les descriptions des données sur le site, une revue peut disposer de prix différents en fonction des
leurs horodatages. Il serait donc normal qu'une revue dispose de plusieurs de plusieurs prix selon différents 
horodatages. Cependant, chaque horodatage pour une revue devrait être unique sinon cela pourrait être considéré comme 
un doublon.

On convertit tout d'abord l'attribut date_stamp en type date.
"""

# %%

price['date_stamp'] = pd.to_datetime(price['date_stamp'], errors='coerce', format='%Y-%m-%d')

# %%

print(f"Valeurs uniques des attributs de price présentant {price.shape[0]} objets:\n"
      f"{get_uniqueness_attributes(price)}")
print(f"Ratio de valeurs vides pour les attributs de price:\n"
      f"{get_ratio_missing_values(price)}")

print(f"Exemples de valeurs possibles pour l'attribut influence_id de price:\n"
      f"{get_unique_values_of_attribute(price, 'influence_id')[1:8]}")
print(f"Valeurs possibles pour l'attribut license de price:\n"
      f"{get_unique_values_of_attribute(price, 'license')}")

# %%
"""
Les attributs influence_id, url et license présentent une majorité de valeurs manquantes, ces attributs semblent donc
peu porteurs d'information.

On s'intéresse principalement aux valeurs de l'attribut price.
"""

# %%

mean_price_per_year = get_mean_price_per_year()

fig, ax = plt.subplots()
plt.bar(range(len(mean_price_per_year)), mean_price_per_year.values())
plt.xticks(range(len(mean_price_per_year)), mean_price_per_year.keys())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title("Moyenne par année des coûts de publications dans les revues")
plt.show()

# %%
"""
Si les deux dernières années (2016, 2017) sont les années où les coûts de publications se sont révélés les plus 
importants, la moyenne du coût de publication de'année 2016 semble être quand même 4 fois plus élévé qu'en 2017.
Les coûts de publication ne semblent pas être stables en fonction des années.
"""

# %%
"""
### Table influence
"""

# %%

print(influence.head())

# %%
"""
journal_name: nom textuel de la revue (semble être un duplicata de l'attribut journal_name de la table journal).
Les valeurs sont textuelles, ne suivant pas de valeurs catégorielles particulière à priori.

issn: identifiant du journal (semble être un duplicata de l'attribut issn de la table journal)
Les valeurs de cet attribut semblent suivre un format particulier tel que: 4 digits - 4 digits

citation_count_sum: indique le nombre de citations total de la revue 

paper_count_sum: indique le nombre de papiers dans lequel la revue est citée

avg_cites_per_paper: indique la moyenne des citations par papier de la revue
Cet attribut semble être un rapport des attributs citation_count_sum et paper_count_sum, permettant de donner un 
résultat plus général sur les citations d'une revue.

proj_ai: information sur le score d'influence des articles de la revue. Celui-ci semble être plus élevé plus la 
moyenne des citations par papier de la revue (correspond à la l'attribut avg_cites_per_paper) est grand.

proj_ai_year: spécification de l'année où l'information sur le score d'influence des articles du journal a été établie


L'attribut issn devrait identifié chaque objet de la table, celui-ci devrait donc être unique.
"""

# %%

influence['proj_ai_year'] = pd.to_datetime(influence['proj_ai_year'], errors='coerce', format='%Y')

# %%

print(f"Valeurs uniques des attributs de influence présentant {influence.shape[0]} objets:\n"
      f"{get_uniqueness_attributes(influence)}")
print(f"Ratio de valeurs vides pour les attributs de influence:\n"
      f"{get_ratio_missing_values(influence)}")

print(f"Valeurs possibles pour l'attribut proj_ai_year de influence:\n"
      f"{get_unique_values_of_attribute(influence, 'proj_ai_year')}")

# %%
"""
L'attribut proj_ai_year ne présentant qu'une seule valeur nous indique que les valeurs de l'attribut proj_ai ont toutes
été établies à la même période, 2015. L'attribut proj_ai_year nous importe donc peu d'information pour chaque objet.
"""

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
Lors de l'importation des données, les indexes de la table étaient fournis. L'unicité des indexes de la table sont à 
vérifier.

On élimine les objets présentant des objets dupliqués sur tous les attributs, au cas où il en existe.

Dans un second temps, comme expliqué dans la question précédente, les objets de la table price se doivent d'être
uniques selon deux attributs, date_stamp et journal_id. Chaque revue peut présenter plusieurs coûts de publication à des
horodatages différents. S'il existe 2 horodatages identiques pour la même revue, alors ce serait des objets considérés
comme dupliqués.
"""

# %%

check = np.logical_not(price.index.duplicated().any())
print(f"Unicité des indexes de la table price: {check}")

nb = len(price)
price = get_df_duplicated_rows_dropped(price)
print(f"Nombre d'objets entièrement identifiques à éliminer dans la table price: {nb - len(price)}")

duplicated_rows = price[price[['date_stamp', 'journal_id']].duplicated(keep=False)]
print(f'Nombre de duplicata selon les attributs date_stamp et journal_id: {len(duplicated_rows)}')

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
arbitraire.
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
Lors de l'importation des données, les indexes de la table étaient fournis. L'unicité des indexes de la table sont à 
vérifier.

On se base sur l'attribut issn qui devrait être unique pour chaque objet de la table, on vérifie son unicité.
"""

# %%

check = np.logical_not(influence.index.duplicated().any())
print(f"Unicité des indexes de la table influence: {check}")

nb = len(influence)
influence = get_df_duplicated_rows_dropped(influence)
print(f"Nombre d'objets entièrement identifiques à éliminer dans la table influence: {nb - len(influence)}")

check = np.logical_not(influence['issn'].duplicated().any())
print(f"Unicité de l'attribut issn dans la table influence: {check}")

# %%
"""
Comme expliqué dans la question précédente, l'attribut avg_cites_per_paper est un résultat du rapport entre 
citation_count_sum et paper_count_sum. Les valeurs de l'attribut avg_cites_per_paper sont donc à vérifier.
"""

# %%

check = influence.apply(
    lambda x: True if x['citation_count_sum'] / x['paper_count_sum'] == x['avg_cites_per_paper'] else False, axis=1)
print(
    f'Rapport citation_count_sum et paper_count_sum est bien égal à avg_cites_per_paper pour tous objets: {check.any()}')

# %%
"""
## Fusion des tables journal, price et influence
Afin de simplifier les opérations, on génère une seule table résumant les différentes informations des trois tables.
On vérifie d'abord si les valeurs des identifiants communs aux différentes tables sont présentes dans les tables à 
fusionner. Pour cela, on vérifie si les valeurs de l'attribut issn de la table influence sont existantes dans l'attribut
issn dans la table journal. Suivant la même idée, on vérifie les valeurs de l'attribut journal_id de price sont 
existantes dans l'attribut issn dans journal. 
"""

# %%

check = influence['issn'].isin(journal['issn']).any()
print(f"Pas de valeur d'issn manquante dans journal par rapport à influence : {check}")

check = price['journal_id'].isin(journal['issn']).any()
print(f"Pas de valeur d'issn manquante dans journal par rapport à price : {check}")

# %%
"""
Les attributs sur lesquels on se base pour la fusion sont effectivement bien représentés dans les tables à fusionner, 
on peut donc envisager la fusion.

On applique maintenant la fusion des trois tables en deux étapes. D'abord, on fusionne la table influence dans la 
table journal. Ensuite, cette table résultante sera fusionnée avec la table price.

Les tables price et journal présentent tous deux un attribut sous le nom 'url', cependant ils ne représentent pas les 
mêmes attributs. Pour la table price, on renommera cet attribut en 'url_author', et pour la table journal, ce sera 
'url_journal'.
"""

# %%

price = rename_df_headers(price, {"journal_id": "issn", "url": "url_author"})
journal = rename_df_headers(journal, {"url": "url_journal"})

temp = pd.merge(journal, influence, on='issn', how='outer')
check = temp.apply(
    lambda x: True if x['journal_name_x'] == x['journal_name_y'] or x['journal_name_y'] is np.nan else False, axis=1)
print(f"Non existence d'aberrances entre les attributs journal_name des tables journal et influence: {check.any()}")

# %%
"""
Les attributs journal_name_x et journal_name_y présentent les mêmes valeurs, on choisit d'éliminer arbitrairement 
l'attribut journal_name_y au profit de journal_name_x.
"""

# %%

temp = temp.drop(columns=['journal_name_y'])
temp = rename_df_headers(temp, {"journal_name_x": "journal_name"})

print(f"Valeurs uniques des attributs de temp présentant {temp.shape[0]} objets:\n"
      f"{get_uniqueness_attributes(temp)}")

data = pd.merge(temp, price, on=['issn'], how='outer')
data = get_df_duplicated_rows_dropped(data)

print(f"Valeurs uniques des attributs de data présentant {data.shape[0]} objets:\n"
      f"{get_uniqueness_attributes(data)}")
print(f"Ratio de valeurs vides pour les attributs de data:\n"
      f"{get_ratio_missing_values(data)}")

# %%
"""
On est effectivement assuré que les valeurs de l'attribut issn de la nouvelle table data sont uniques.
"""

# %%
"""
## B. Y-a-t-il une corrélation entre les catégories de journaux (attribut category) et les coûts de publication 
(attribut price)? Justifier la réponse.

Afin de déterminer s'il existe une corrélation entre les catégories et l'attribut prix, on s'intéresse à chaque 
catégorie une à une et sa corrélation propre avec l'attribut prix. 

Comme précisé dans la question 1, chaque objet peut avoir plusieurs valeurs de catégories. On observe différents 
séparateurs ('|', 'and', '.') entre les différentes valeurs de catégories.
Après séparation des différentes catégories, on les convertit ensuite en one hot multivaleurs.

Ainsi, on peut calculer la corrélation catégorie par catégorie avec l'attribut prix. Pour cela, on ne considère que les 
objets présentant la catégorie testée et les valeurs de prix associées.
"""

# %%

cat_labelled_data = data[data['category'].notna()]
cat_data_to_predict = data[data['category'].isna()]

# %%
# TODO: category reductions

cat_labelled_data['category'] = cat_labelled_data['category'].str.replace(r'[\.\|&] | [\.\|&] | and ', '.', regex=True)
category_dummies = cat_labelled_data['category'].str.get_dummies(sep='.')
category_dummies_prefix = category_dummies.add_prefix('category_')
print(f'Nombre de catégories après séparation: {category_dummies.shape[1]}')

# %%

cat_labelled_data = pd.concat([cat_labelled_data, category_dummies_prefix], axis=1)

# %%

categories_correlation = {}

for header in category_dummies_prefix.columns:
    corr = cat_labelled_data[header].corr(cat_labelled_data['price'])
    if abs(corr) > 0.1:
        categories_correlation[header] = corr

fig, ax = plt.subplots()
plt.bar(categories_correlation.keys(), categories_correlation.values())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title(f'Catégories présentant des corrélations fortes avec\nl\'attribut prix et leurs valeurs')
plt.show()

# %%
"""
On remarque que certaines catégories présentent effectement une légère corrélation avec l'attribut prix.
(Les catégories présentant une corrélation inférieures à 0.1 ne sont pas incluses dans le graphe)
Les catégories présentant la plus forte corrélation sont 'cell biology' et 'molecular'.
Cette corrélation remarquée est faible mais ne peut être négligeable pour certaines catégories.
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
Cependant, ces deux attributs sont des données textuelles non catégorielles très inconsistantes. Afin de pouvoir en 
sortir quelque chose, une solution serait de calculer la similarité de ces valeurs avec le nom de chacune des catégories
que l'on a pu déterminer dans la question précédente. On utilise alors la classe Sequence Matcher sur journal_name et 
pub_name avec le nom de chacune des catégories et inscrit chacun des scores au sein d'un nouvel attribut pour la table 
cat_labelled_data.

A la question précédente, on a pu trouver une corrélation légère entre certaines catégories et l'attribut prix. On peut 
donc envisager de prendre en compte le paramètre prix.

Les informations de citation du journal pourraient également se révéler porteuses d'informations, on considère alors 
l'utilisation de l'attribut avg_cites_per_paper qui résume ces informations.

On peut considérer également que certaines catégories ont plus d'influence de manière globale que d'autres, on choisit
alors d'utiliser également l'attribut proj_ai.
"""

# %%

cat_labelled_data = cat_labelled_data[cat_labelled_data['price'].notna()]
headers = ['avg_cites_per_paper', 'proj_ai', 'price']

# élimination des objets présentant des valeurs nulles dans les attributs d'intérêts
cat_labelled_data = cat_labelled_data.dropna(axis=0, subset=headers)

# %%

# création des attributs correspondant au score de journal_name et pub_name avec chaque catégorie
for header in tqdm(category_dummies.columns):
    cat_labelled_data['jn_' + header] = cat_labelled_data.apply(partial(get_score_sequence_matching, c1='journal_name',
                                                                        category=header), axis=1)
    cat_labelled_data['pn_' + header] = cat_labelled_data.apply(partial(get_score_sequence_matching, c1='pub_name',
                                                                        category=header), axis=1)

# %%

cat_labelled_data = cat_labelled_data.drop(columns=['category'])

# %%

# extraction des noms des attributs score pour journal_name et pub_name
jn_sm_headers = cat_labelled_data.filter(like='jn_').columns.to_list()
pn_sm_headers = cat_labelled_data.filter(like='pn_').columns.to_list()

# ajout des attributs score aux attributs d'intérêts pour le modèle
attributes_of_interest = headers
attributes_of_interest.extend(jn_sm_headers)
attributes_of_interest.extend(pn_sm_headers)

# %%
"""
### Entrainement
On s'intéresse à un système de classification ayant la capacité de pouvoir préduire des labels multiples.
Pour cela, on utilise la méthode MultiOutputClassifier de sklearn afin qui consiste à adapter un classificateur par
cible.
A partir de là, on a pu essayer plusieurs types de classification, la meilleure s'est révélée être un modèle Random
Forest. Etant donné que l'on dispose de données numériques et catégorielles, la performance d'un Random Forest n'est pas
étonnante. Aussi, le fait qu'un modèle de ce type ait une capacité à ne pas sur-apprendre de trop permet de généraliser 
bien sur nos données. 

Le code suivant sert à faire une recherche d'hyperparamètres (succinte, pour le confort de la compilation) sur un 
classification Random Forest.
"""

# %%

clfs = {'RandomForestClassifier': RandomForestClassifier()}

best_model = {'name': '', 'score': 0, 'model': None}
for name, clf in clfs.items():
    for i in range(14, 17):
        X_train, X_test, y_train, y_test = train_test_split(cat_labelled_data[headers],
                                                            cat_labelled_data[category_dummies_prefix.columns],
                                                            test_size=0.2, random_state=42)
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
Exemple de compilation du code de la cellule précédente:

Modèle RandomForestClassifier 14
-- Entrainement
Score d'entraînement: 0.9972086531751571
-- Test
Score de test: 0.7383309759547383

Modèle RandomForestClassifier 15
-- Entrainement
Score d'entraînement: 0.9986043265875785
-- Test
Score de test: 0.751060820367751

Modèle RandomForestClassifier 16
-- Entrainement
Score d'entraînement: 0.9986043265875785
-- Test
Score de test: 0.7411598302687411

Le modèle présentant le meilleur score est RandomForestClassifier 15 avec 0.751060820367751

"""

# %%
"""
On conclut ainsi que le classificateur Random Forest ayant une profondeur maximale de 15 présente des résultats se
trouve être le plus performant.
Aussi, on se trouve en présence de résultats très performants, étant donné qu'on dispose de 88 labels à prédire, soient
les catégories.
"""

# %%
"""
### Prédictions
On effectue maintenant les prédictions sur les objets présentant les catégories manquantes.
"""

# %%

headers = ['avg_cites_per_paper', 'proj_ai', 'price']
cat_data_to_predict = cat_data_to_predict.dropna(axis=0, subset=headers)

# %%

# Comme lors de la partie entrainement, on calcule le score de Sequence Matcher entre les attributs journal_name et
# pub_name et chacune des catégories
for header in tqdm(category_dummies.columns):
    cat_data_to_predict['jn_' + header] = cat_data_to_predict.apply(partial(get_score_sequence_matching,
                                                                            c1='journal_name', category=header), axis=1)
    cat_data_to_predict['pn_' + header] = cat_data_to_predict.apply(partial(get_score_sequence_matching, c1='pub_name',
                                                                            category=header), axis=1)

# %%

clf = best_model.get('model')  # Random Forest with max_depth=15
predictions = pd.DataFrame(clf.predict(cat_data_to_predict[attributes_of_interest]))
predictions.columns = category_dummies_prefix.columns

# %%

# On répertorie le nombre de catégories prédites
count_categories = {}
for header in predictions.columns:
    nb = predictions[header].sum()
    if nb >= 1:
        count_categories[header] = nb

# %%

fig, ax = plt.subplots()
plt.bar(count_categories.keys(), count_categories.values())
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.title(f'Somme des catégories prédites par le modèle')
plt.show()

# %%

# Ajout des prédictions des catégories à leurs objets respectifs dans la table cat_data_to_predict
predictions = predictions.set_index(cat_data_to_predict.index.copy())
for header in predictions.columns:
    cat_data_to_predict[header] = predictions[header]

# %%
"""
# Question 3: Régression-clustering
## A. Supprimer tous les attributs ayant plus de 50% de données manquantes.

On établit les attributs présentant 50% de données manquantes à éliminer selon les 3 tables originales (journal, price
et influence).

Cependant, on utilise nos données déjà travaillées qui sont dans les tables cat_labelled_data et cat_data_to_predict.
Aussi, au vu des bons résultats de prédictions du modèle de classification pour les catégories, on peut se permettre 
d'utiliser les objets prédits pour la suite du travail. 

On établit alors une nouvelle table, nommée data, présentant l'ensemble des données, originales et prédites, et on peut
y éliminer les attributs présentant plus de 50% de valeurs manquantes.
"""

# %%

print(f'Attributs à éliminer de la table journal: {get_empty_attribute_to_remove(journal)}')
print(f'Attributs à éliminer de la table price: {get_empty_attribute_to_remove(price)}')
print(f'Attributs à éliminer de la table influence: {get_empty_attribute_to_remove(influence)}')

# %%

# cat_data_to_predict = cat_data_to_predict.drop(columns='category')
data = cat_labelled_data.append(cat_data_to_predict, sort=False)
data = data.drop(columns=['url_journal', 'influence_id', 'url_author', 'license'])

# %%
"""
## B. Construire un modèle pour prédire le coût actuel de publication (attribut «price») à partir des autres attributs 
(cela inclut la sélection d’attributs informatifs, le choix et le paramétrage d’un modèle de régression, le calcul du 
score du modèle, l’application du modèle pour prédire les coûts).Justifier les choix effectués.
Lister les 10 revues qui s’écartent le plus (en + ou -) de la valeur prédite.
 
 
L'attribut date_stamp établissant la date à laquelle le coût de publication a été mesuré semble intéressant. On décide 
de garder seulement l'année car une précision plus importante semble peu pertinente.

Comme vu précédemment l'attribut price présente seulement une faible corrélation aux catégories d'une revue, cependant
certaines catégories sortaient du lot. Les catégories sous la forme de vecteur one hot multivaleurs sont donc à 
envisager.

L'information d'un journal sur son hybridicité est également un facteur important sur son coût de publication. 

L'attribut avg_cites_per_paper, présentant le rapport entre les attributs citation_count_sum et paper_count_sum, révèle
à quel point la revue est cité par papier. Il ne serait pas étonnant que cette information soit liée au prix de 
publication d'articles au sein de la revue. 
Les attributs citation_count_sum et paper_count_sum ne sont pas nécessaires car résumés dans l'attribut 
avg_cites_per_paper.

L'attribut proj_ai présente l'influence des articles d'une revue est également important et on pourrait sans soucis 
imaginer que plus la valeur d'influence est élevée, plus le coût de publication dans une revue serait important.
Néanmoins, l'attribut proj_ai_year, présentant toujours la même valeur (2015) ne nous apporterait aucune information 
sur le jeu de données. 

### Construction et estimation des performances du modèle
"""

# %%

data['year_price'] = pd.DatetimeIndex(data['date_stamp']).year

# %%

attributes_of_interest = ['year_price', 'avg_cites_per_paper', 'proj_ai', 'is_hybrid', 'price']
attributes_of_interest.extend(category_dummies_prefix.columns)

# %%

price_labelled_data = data[attributes_of_interest]

# %%

best_model = {'depth': 0, 'score': 0, 'model': None}
for d in range(15, 18):
    X_train, X_test, y_train, y_test = train_test_split(price_labelled_data.drop(columns='price'),
                                                        price_labelled_data['price'],
                                                        test_size=0.2, random_state=42)
    print(f'Random Forest max profondeur={d}')
    regr = RandomForestRegressor(max_depth=d, n_estimators=250, n_jobs=-1)
    print('-- Entrainement')
    regr.fit(X_train, y_train)
    train_score = regr.score(X_train, y_train)
    print(f'Score d\'entraînement: {train_score}')

    print('-- Test')
    test_predictions = regr.predict(X_test)
    test_score = regr.score(X_test, y_test)
    print(f'Score de test: {test_score}')
    if test_score > best_model.get('score'):
        best_model['depth'], best_model['score'], best_model['model'] = d, test_score, regr

print(f"Le modèle présentant le meilleur score est {best_model.get('depth')} avec {best_model.get('score')}")

# %%
"""
Le modèle Random Forest avec un profondeur maximale de 16 se révèle être le meilleur avec un score de test aux alentours
de 78%. Cela se révèle être un très bon modèle.
"""

# %%
"""
### Application du modèle
On applique ce modèle sur l'ensemble de nos données (entrainement et test) afin de déterminer quels sont les objets dont
les prédictions sont les moins bonnes.
"""

# %%

predictions = pd.DataFrame(regr.predict(price_labelled_data.drop(columns='price')))
predictions = predictions.set_index(data.index.copy())

# %%

difference_pred_real = dict()
for index, p in predictions.iterrows():
    difference_pred_real[data['journal_name'][index]] = abs(p[0] - data['price'][index])

# %%

print(f'Nom des 10 revues présentant les plus gros écarts entre leur prédiction de coût et la réalité:\n'
      f'\nNom: différence')
worst_predictions = np.array(heapq.nlargest(10, difference_pred_real, key=difference_pred_real.get))
worst_predictions_values = []
for p in worst_predictions:
    worst_predictions_values.append(difference_pred_real.get(p))
    print(f'{p} : {difference_pred_real.get(p)}')

worst_predictions = np.vstack([worst_predictions, worst_predictions_values])

# %%
"""
Les 10 revues où les prédictions s'éloignent le plus de la réalité présentent des différences de prédictions très 
importantes (+ de $2000). Cela peut s'expliquer assez simplement avec le fait que de nombreuses valeurs de coûts de 
publication sont à $0 alors qu'un nombre également important sont à $3000.
Ce grand écart n'est donc pas très révélateur.
"""

# %%
"""
## C. Construire un modèle pour grouper les revues selon le coût actuel de publication (attribut "price") et le score
d'influence (attribut "proj_ai") (cela inclut la détermination du nombre de clusters, le choix et le paramétrage d'un
modèle de clustering, l'application du modèle pour trouver les clusters). Justifier les choix.

Etant donné que les mesures de distance vont être importantes pour déterminer les clusters, il serait pertinent de 
normaliser et centraliser les données.
Nos données sont très regroupées en un bloc et présente quelques données que l'on pourrait qualifier de données 
aberrantes.
On essaie alors différentes méthodes de clustering qui se révèlent être performantes sur des données peu séparées en 
clusters bien définis, soient Agglomerative Clustering, DBSCAN et Gaussian Mixture. 
Aussi, on essaie également KMeans afin pour se donner une référence, néanmoins celui-ci devrait être moins bon que les
autres.
"""

# %%

attributes_of_interest = ['price', 'proj_ai']
data_for_clustering = data[attributes_of_interest]
norm_data_for_clustering = StandardScaler().fit_transform(data_for_clustering)

# %%

estimators = {'K Means 4 clusters': KMeans(n_clusters=4, random_state=42),
              'Agglomerative Clustering 4 clusters, ward': AgglomerativeClustering(n_clusters=4, linkage='ward'),
              'DBSCAN': DBSCAN(),
              'GaussianMixture 3 clusters, diag': GaussianMixture(n_components=3, covariance_type='diag')}

for name, estimator in estimators.items():
    plt.figure()
    y_pred = estimator.fit_predict(norm_data_for_clustering)
    plt.scatter(data_for_clustering['price'], data_for_clustering['proj_ai'], c=y_pred)
    plt.title(name)
    plt.show()

    if name == 'DBSCAN':
        best_estimator_pred = y_pred

# %%
"""
Tout d'abord, on remarque que K Means (4 clusters) et Agglomerative Clustering (4 clusters) effectuent quasiment 
le même regroupement. Ces deux méthodes ne permettent pas de vraiment faire ressortir les clusters tels qu'on les 
voient et les divisent. 

Gaussian Mixture (3 clusters) présente toutes les 'données extrêmes' au sein d'un même cluster, puis trouve un autre
cluster près de l'origine. Le cluster des 'données extrêmes' ne semble pas pertinent. 

La méthode DBSCAN représente les données extrêmes comme aberrantes et spécifie 3 clusters. Un des clusters regroupe la
majorité des données qui sont très regroupés. Les 2 autres clusters sont des données un peu plus éparses avec quelques
points mais néanmoins non négliables. On peut donc conclure que DBSCAN et sa représentation en 3 clusters donnent des
clusters probants.
"""

# %%
"""
## D. Présenter des statistiques descriptives des clusters obtenus, et lister les revues du meilleur cluster en termes 
en termes de rapport moyen: score d'influence / coût de publication. 
"""

# %%

data_for_clustering = data_for_clustering.copy()
data_for_clustering['cluster_predicted'] = best_estimator_pred

# %%

stats_clusters = dict()
for c in data_for_clustering['cluster_predicted'].unique():
    print(f'\nCluster {c}:')
    temp = data_for_clustering[data_for_clustering['cluster_predicted'] == c]
    print(f"{temp[['price', 'proj_ai']].describe()}")

    if temp['price'].mean() == 0:
        ratio = temp['proj_ai'].mean() / 0.00001
    else :
        ratio = temp['proj_ai'].mean() / temp['price'].mean()
    print(f'Rapport moyen entre le score d\'influence et les coûts de publication: {ratio}')

# %%
"""
Le cluster présentant le meilleur rapport moyen entre le score d'influence et les coûts de publication est celui nommé
1, présentant des coûts de publication nuls et un score d'influence suffisamment élevé.
"""

# %%

print(f"Liste des revues dans le cluster ayant le meilleur rapport coûts de publication et score d'influence:"
      f"{data[data_for_clustering['cluster_predicted'] == 1][['journal_name', 'pub_name']]}")

