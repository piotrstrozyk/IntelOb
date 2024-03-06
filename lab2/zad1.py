import pandas as pd
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
# missing_values = df.isnull().sum()
# print(missing_values)

# #statystyki
# stats = df.describe()
# print(stats)

#b
df['sepal.length'] = pd.to_numeric(df['sepal.length'], errors='coerce')
df['sepal.width'] = pd.to_numeric(df['sepal.width'], errors='coerce')

# Function to fix values within a specified range for each column
def fix_range(df):
    for column in df.columns:
        if df[column].dtype in [int, float]:
            median = df[column].median()
            df[column] = df[column].apply(lambda x: median if not (0 < x < 15) else x)
    return df

# Apply the function to fix values within the specified range for each column
df_fixed = fix_range(df)
print(df_fixed)



types = df.dtypes
print(types)