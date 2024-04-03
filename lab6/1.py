import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')
df = df.drop(df.columns[0], axis=1)

# items = set()

# for col in df:
#     items.update(df[col].unique())

# print(items)

# itemset = set(items)
# encoded_vals = []
# for index, row in df.iterrows():
#     rowset = set(row) 
#     labels = {}
#     uncommons = list(itemset - rowset)
#     commons = list(itemset.intersection(rowset))
#     for uc in uncommons:
#         labels[uc] = 0
#     for com in commons:
#         labels[com] = 1
#     encoded_vals.append(labels)
# encoded_vals[0]
# ohe_df = pd.DataFrame(encoded_vals)

# print(ohe_df)   


data = pd.get_dummies(df, columns=['Class', 'Sex', 'Age', 'Survived'])

#print(data.head())

freq = apriori(data, min_support=0.005, use_colnames=True, verbose=1)

#print(freq.head(20))

rules = association_rules(freq, metric="confidence", min_threshold=0.8)
rules = rules.sort_values('conviction', ascending=False)
#print(rules.head(20))

# Filtrowanie reguł dotyczących przeżywalności
survival_rules = rules[rules['consequents'].apply(lambda x: 'Survived_Yes' in str(x))]

# Sortowanie reguł wg ufności
survival_rules = survival_rules.sort_values('confidence', ascending=False)

# Wyświetlenie najciekawszych reguł
print(survival_rules)
