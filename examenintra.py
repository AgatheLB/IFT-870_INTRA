# %%

# leba3207

import numpy as np
import pandas as pd

# %%

data_folder = 'data/'

journal_file = 'api_journal11-13-17.csv'
price_file = 'api_price11-13-17.csv'
influence_file = 'estimated-article-influence-scores-2015.csv'

journal = pd.read_csv(data_folder + journal_file, sep=',', encoding='latin1')
price = pd.read_csv(data_folder + price_file, sep=',')
influence = pd.read_csv(data_folder + influence_file, sep=',')


# %%

def get_uniqueness_attributes(table):
    return table.nunique(axis=0)


def get_ratio_missing_values(table):
    return table.isnull().sum() * 100 / len(table)


def lowercase_columns(table, headers):
    for header in headers:
        table[header] = table[header].str.lower()

# %%

"""
# Question 1: Exploration-Description
## Présenter une description de chacun des attributs des 3 tables, avec des graphiques pour la visualisation des 
statistiques descriptives au besoin.

### Table journal
"""

# %%

print(journal.head())

# %%

"""
issn: identifiant du journal

journal_name: nom textuel du journal

pub_name: nom de l'éditeur du journal

is_hybrid: valeur booléenne indiquant si le journal possède son propre éditeur à son nom (??)

category: renseigne sur les catégories de sujet abordés par le journal 
Il peut exister de multiples valeurs pour chaque objet, leur séparation ne semble d'ailleurs pas consistante.

url: indique l'adresse url du site du journal


Les attributs journal_name, pub_name et category étant des données textuelles très inconsistantes, je décide avant tout
traitement et étude supplémentaire de transformer les valeurs en minuscule pour limiter au maximum l'inconsistence.
"""

# %%

lowercase_columns(journal, ['journal_name', 'pub_name', 'category'])

# %%

print(f'Valeurs uniques des attributs de journal:\n{get_uniqueness_attributes(journal)}')
print(f'Ratio de valeurs vides pour les attributs de journal:\n{get_ratio_missing_values(journal)}')

# %%

"""
On remarque qu'il existe uniquement deux valeurs pour l'attribut is_hybrid (soit 1 soit 0).
Les attributs category, url présentent un nombre conséquent de valeurs manquantes.
"""

# %%


# %%

"""
# Question 2: Prétraitement-Représentation
## A. Effectuer un prétraitement des données pour supprimer les duplications et corriger les incohérences s’il y en a.
"""

# TODO: separator in journal/category : and . |
