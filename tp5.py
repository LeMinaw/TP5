# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

def __EXO__(number):
    print(f"\n{12*'='} EXO N°{number} {12*'='}\n")



__EXO__(1.1)
 
t = "26-oct. 20:53"

x, yz = t.split("-") # "26", "oct. 20:53"
y, z = yz.split(" ") # "oct.", "20:53"



__EXO__(1.2)

def parse_date(date_str):
    mois = { # Dictionnaire indexé selon les codes de mois abrégés
        'août': '08',
        'sept.': '09',
        'oct.': '10',
        'nov.': '11',
        'déc.': '12',
    }
    j, mh = date_str.split("-") # j contient le jour, mh le code du mois et l'heure
    m, h = mh.split(" ") # m contient le code du mois, h contient l'heure

    # Accède au dictionnaire par code de mois pour obtenir le numéro
    # du mois, puis concatène les éléments
    return '2019-' + mois[m] + '-' + j + ' ' + h

    # Autre version équivalente avec une f-string
    return f"2019-{mois[m]}-{j} {h}" 



__EXO__(2.1)

a = [['a', 2, 3, 4], ['a', 3, 5, 2], ['b', 1, 2, 3], ['c', 3, 2, 1], ['b', 0, 1, 2]]
l = ['A', 'B', 'C', 'D']
df = DataFrame(a, columns=l)

print(f"DataFrame :\n{df}")
print(f"Transposée :\n{df.T}")



__EXO__(2.2)

df['A']
# 0    a
# 1    a
# 2    b
# 3    c
# 4    b
# Name: A, dtype: object

df.loc[0]
# A    a
# B    2
# C    3
# D    4
# Name: 0, dtype: object

df.loc[[1, 2], ['A', 'C']]
#    A  C
# 1  a  5
# 2  b  2

df.loc[:, ['A', 'C']]
#    A  C
# 0  a  3
# 1  a  5
# 2  b  2
# 3  c  2
# 4  b  1

df.loc[1, 'C'] # 5

df.at[1, 'C'] # 5

df.iloc[1]
# A    a
# B    3
# C    5
# D    2
# Name: 1, dtype: object

df.iloc[1:3, [0, 2]]
#    A  C
# 1  a  5
# 2  b  2

df.iloc[:, 2:4]
#    C  D
# 0  3  4
# 1  5  2
# 2  2  3
# 3  2  1
# 4  1  2

df.iloc[1, 2] # 5

df.iat[1, 2] # 5

# loc:  Renvoie une plage de valeurs par lignes et colomnes par label
# at:   Renvoie un seul élément à une ligne et colomne donnée par label
# iloc: Renvoie une plage de valeurs par lignes et colomnes par index entier
# iat:  Renvoie un seul élément à une ligne et colomne donnée par index entier



__EXO__(2.3)

for x in df.loc[:, ['A']], df.loc[:, 'A']:
    print(f"Valeur :\n{x}\nType : {type(x)}")

# df.loc[:, ['A']] renvoie une nouvelle DataFrame
# df.loc[:, 'A'] renvoie une série de la DataFrame d'origine



__EXO__(2.4)

print("Infos :")
df.info(memory_usage=True)



__EXO__(2.5)

print(f"Tête :\n{df.head(2)}")
print(f"Queue :\n{df.tail(2)}")

# DataFrame.head() et DataFrame.tail() retournent les
# cinq premiers et derniers éléments par défaut.



__EXO__(2.6)

print(df.aggregate(('count', 'sum', 'min', 'max', 'mean', 'var', 'std')))
print(df.aggregate(('count',), axis='columns'))



__EXO__(2.7)

print([x for x in df]) # ['A', 'B', 'C', 'D']
print([x.A for x in df.itertuples()]) # ['a', 'a', 'b', 'c', 'b']



__EXO__(2.8)

df[df['B'] > 2]
#    A  B  C  D
# 1  a  3  5  2
# 3  c  3  2  1

df[(df['C'] > 2) & - (df['B'] < 3)]
#    A  B  C  D
# 1  a  3  5  2

df[df['A'].isin(['a', 'c', 'm'])]
#    A  B  C  D
# 0  a  2  3  4
# 1  a  3  5  2
# 3  c  3  2  1

df.query('B > 2 and C < 10')
#    A  B  C  D
# 1  a  3  5  2
# 3  c  3  2  1

np.where(df['B'] < 3)[0]
# [0 2 4]



__EXO__(2.9)

for func in ('min', 'max', 'mean', 'median', 'std', 'var', 'sum', 'prod'):
    res = eval(f"df.{func}()")
    print(f"{func.capitalize()} :\n{res}")



__EXO__(2.10)

print(df.groupby('B').mean())



__EXO__(3.1)

# Voir fichier cards.py joint



__EXO__(3.2)

# Le fichier contient les données suivantes, en colonnes :
# Nom | Identifiant | Nom de l'annonce | Prix de vente | Date de vente | Provenance



__EXO__(3.3)

cards_data = pd.read_excel("données_cartes.xlsx", dtype={"Prix de vente": np.float64})
print(cards_data)



__EXO__(3.4)

mean_prices = cards_data.groupby("Nom").mean()["Prix de vente"]
print(mean_prices)



__EXO__(3.5)

mean_prices.plot.bar()
plt.show()
mean_prices.sort_values().plot.line()
plt.show()



__EXO__(3.6)

print(f"Moyenne : {mean_prices.mean()}")
print(f"Écart-type : {mean_prices.std()}")



__EXO__(3.7)

# Series.argmin et Series.argmax sont dépréciées en faveur de Series.idxmin 
# et Series.idxmax. Ces deux fonctions retournent le label des valeurs
# maximales et minimales de la série. Dans le futur, le comportement de
# Series.argmin et Series.argmax sera de retourner l'index numérique des
# valeurs minimales et maximales à la place du label. C'est le comportement
# actuel des méthodes Series.values.argmax et Series.values.argmin.

print(f"Minimum : {mean_prices.idxmin()} (position {mean_prices.values.argmin()})")
print(f"Maximum : {mean_prices.idxmax()} (position {mean_prices.values.argmax()})")



__EXO__(4.1)

def convert_date(date_str):
    return pd.to_datetime(parse_date(date_str))

cards_data = pd.read_excel(
        "données_cartes.xlsx",
        dtype={"Prix de vente": np.float64},
        converters={"Date de vente": convert_date}
).set_index("Date de vente")
date_prices = cards_data["Prix de vente"]

plt.figure(figsize=(9, 6))

plt.subplot(211)
date_prices.plot(grid=True)
plt.xlabel("Date de vente")
plt.ylabel("Prix de vente")

plt.subplot(212)
date_prices.hist(range=(0, 100), bins=70, color='yellow', edgecolor='red')
plt.xlabel("Prix de vente")
plt.ylabel("Fréquence")

plt.show()
plt.close()



__EXO__(4.2)

sept_prices = cards_data["2019-09"]["Prix de vente"]
sept_nov_prices = cards_data["2019-09":"2019-11"]["Prix de vente"]

plt.figure(figsize=(9, 6))

plt.subplot(211)
sept_prices.plot(grid=True)
plt.gca().set_title("Septembre 2019")
plt.xlabel('')

plt.subplot(212)
sept_nov_prices.plot(grid=True)
plt.gca().set_title("De septmbre à novembre 2019")
plt.xlabel('')

plt.show()
plt.close()

dates = (
    ("2019-09", "2019-09"),
    ("2019-09", "2019-11")
)
for start, end in dates:
    data = cards_data[start:end].reset_index()
    best = data.loc[data["Prix de vente"].idxmax()]
    print(f"Meilleure vente sur la plage {start}->{end} :\n{best}\n")



__EXO__(4.3)

prices = cards_data["2019"]["Prix de vente"]
month_means = prices.resample('M').mean()
week_means = prices.resample('W').mean()
month_means.plot.line()
week_means.plot.line()
plt.show()
plt.close()



__EXO__(4.4)

week_prices = prices.resample('W')
print(week_prices.agg(('mean', 'std', 'min', 'max')))



__EXO__(4.5)

mean_prices = cards_data.groupby("Nom").mean()["Prix de vente"]
sells = cards_data.groupby("Nom").size()

plt.scatter(mean_prices, sells)
plt.xlabel("Prix moyen")
plt.ylabel("Nombre de ventes")
plt.show()
plt.close()


__EXO__(4.6)

mean_prices_ln = mean_prices.apply(np.log)
sells_ln = sells.apply(np.log)
sells_ln_fit = np.polyfit(mean_prices_ln, sells_ln, 1)
sells_ln_regression = np.poly1d(sells_ln_fit)

plt.scatter(mean_prices_ln, sells_ln)
plt.plot(mean_prices_ln, sells_ln_regression(mean_prices_ln))
plt.xlabel("Prix moyen (ln)")
plt.ylabel("Nombre de ventes (ln)")
plt.show()
plt.close()

print(f"Les coeficients de la régression sont {sells_ln_fit} au premier degré.")
