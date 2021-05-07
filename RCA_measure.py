import numpy as np
import pandas as pd
from metric_learn import RCA
from metric_learn import MMC
from metric_learn import RCA
import seaborn as sns
import matplotlib.pyplot as plt

df_actions = pd.read_csv('data/df_ohe_groupby.csv', delimiter=",").sort_values('tdl_libelle_nomemclature')
df_pairs = pd.read_csv('data/pairs_classe.csv', delimiter=";")
df_benef = pd.read_csv('data/interest_benef.csv', delimiter=";")
print("Input shape", df_actions.to_numpy().shape)

X = df_actions.drop('tdl_libelle_nomemclature', axis=1).to_numpy()
pairs_indices = []
y_pairs = []

for pair in df_pairs.to_numpy():
    pairs_indices.append([np.argmax(df_actions['tdl_libelle_nomemclature'] == pair[0]), np.argmax(df_actions['tdl_libelle_nomemclature'] == pair[1])])
    y_pairs.append(pair[2] if pair[2] == 1 else -1)


pairs_indices = np.array(pairs_indices)
y_pairs = np.array(y_pairs)
print(pairs_indices.shape, y_pairs.shape)

mmc = MMC(preprocessor=X)
mmc.fit(pairs_indices, y_pairs)

#rca = RCA(preprocessor=X)
#rca.fit(pairs_indices, y_pairs)

d = mmc.get_metric()
print('--- Distances matrix ---')
with open("dists.txt", mode="w") as out:
    for i in range(len(df_actions)):
        for j in range(len(df_actions)):
            if i == j:
                print(0.0, end=" ", file=out)
            else:
                #print(X[i].shape, X[j].shape)
                print(d(X[i], X[j]), end=" ", file=out)
        print(file=out)



# Correlations
corr = df_actions.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,  square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.savefig("data/truc.png")