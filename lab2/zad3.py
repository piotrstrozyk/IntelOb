import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv("iris.csv")

# Select variables
X = df[['sepal.length', 'sepal.width']]
y = df['variety']

# Original Data Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
for species in y.unique():
    plt.scatter(X.loc[y == species, 'sepal.length'], X.loc[y == species, 'sepal.width'], label=species)
plt.title("Original Data")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

# Min-Max Normalization
X_normalized = (X - X.min()) / (X.max() - X.min())
plt.subplot(1, 3, 2)
for species in y.unique():
    plt.scatter(X_normalized.loc[y == species, 'sepal.length'], X_normalized.loc[y == species, 'sepal.width'], label=species)
plt.title("Min-Max Normalized Data")
plt.xlabel("Sepal Length (normalized)")
plt.ylabel("Sepal Width (normalized)")
plt.legend()

# Z-score Scaling
X_scaled = (X - X.mean()) / X.std()
plt.subplot(1, 3, 3)
for species in y.unique():
    plt.scatter(X_scaled.loc[y == species, 'sepal.length'], X_scaled.loc[y == species, 'sepal.width'], label=species)
plt.title("Z-score Scaled Data")
plt.xlabel("Sepal Length (scaled)")
plt.ylabel("Sepal Width (scaled)")
plt.legend()

plt.tight_layout()
plt.show()