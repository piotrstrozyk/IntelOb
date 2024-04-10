import pandas as pd # type: ignore
import numpy as np # type: ignore
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')
df = df.drop(df.columns[0], axis=1)

data = pd.get_dummies(df, columns=['Class', 'Sex', 'Age', 'Survived'])

#print(data.head())

freq = apriori(data, min_support=0.005, use_colnames=True, verbose=1)

#print(freq.head(20))

rules = association_rules(freq, metric="confidence", min_threshold=0.8)
rules = rules.sort_values('confidence', ascending=False)
print(rules)

# Filtrowanie reguł dotyczących przeżywalności
survival_rules = rules[rules['consequents'].apply(lambda x: ('Survived_Yes' in str(x) or 'Survived_No' in str(x)) and 'Age_Adult' not in str(x))]

# Sortowanie reguł wg ufności
survival_rules = survival_rules.sort_values('confidence', ascending=False)

# Wyświetlenie najciekawszych reguł
print(survival_rules)
