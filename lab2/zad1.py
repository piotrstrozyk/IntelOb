import pandas as pd
from difflib import get_close_matches
df = pd.read_csv("iris_with_errors.csv")
#print(df)
#print(df.values)

#wszystkie wiersze, kolumna nr 0
# print(df.values[:, 0])
# #wiersze od 5 do 10, wszystkie kolumny
# print(df.values[5:11, :])
# #dane w komórce [1,4]
# print(df.values[1, 4])

#(a
#brakujące
missing_values = df.isnull().sum()
print(missing_values)

#statystyki
stats = df.describe()
print(stats)

#b
df['sepal.length'] = pd.to_numeric(df['sepal.length'], errors='coerce')
df['sepal.width'] = pd.to_numeric(df['sepal.width'], errors='coerce')

def fix_range(df):
    for column in df.columns:
        if df[column].dtype in [int, float]:
            median = df[column].median()
            df[column] = df[column].apply(lambda x: median if not (0 < x < 15) else x)
    return df


df_fixed = fix_range(df)
print(df_fixed.values[:, 0:5])



types = df.dtypes
print(types)

#c
unique_varieties = df['variety'].unique()

# Wyświetlenie unikalnych wartości
print("Unikalne wartości w kolumnie 'variety':", unique_varieties)

data = pd.read_csv('iris.csv')

# Lista poprawnych nazw gatunków
correct_varieties = ["Setosa", "Versicolor", "Virginica"]

# Sprawdzenie i poprawa błędnych nazw gatunków
for index, row in data.iterrows():
    variety = row['variety']
    if variety not in correct_varieties:
        closest_match = get_close_matches(variety, correct_varieties)
        if closest_match:
            data.at[index, 'variety'] = closest_match[0]

# Zapisanie zmodyfikowanych danych do pliku CSV
data.to_csv('iris_corrected.csv', index=False)