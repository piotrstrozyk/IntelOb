from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
print(X.head())
pca_iris = PCA(n_components=3).fit(iris.data)
print(pca_iris)
print(pca_iris.explained_variance_ratio_)
print(pca_iris.components_)
print(pca_iris.transform(iris.data))

import numpy as np

# Obliczenie sumy wyjaśnionej wariancji dla każdej liczby zachowanych komponentów
cumulative_variance_ratios = np.cumsum(pca_iris.explained_variance_ratio_)

# Obliczenie sumy wyjaśnionej wariancji dla różnych ilości zachowanych komponentów
total_variance = np.sum(pca_iris.explained_variance_ratio_)

# Obliczenie sumy wyjaśnionej wariancji dla różnych ilości zachowanych komponentów w odwróconym porządku
reversed_cumulative_variance_ratios = np.cumsum(pca_iris.explained_variance_ratio_[::-1])[::-1]

# Wypisanie sumy wyjaśnionej wariancji dla różnych ilości zachowanych komponentów
for i, (variance_ratio, reversed_variance_ratio) in enumerate(zip(cumulative_variance_ratios, reversed_cumulative_variance_ratios)):
    print(f"Suma wyjaśnionej wariancji dla {i+1} zachowanych komponentów: {variance_ratio:.4f} ({reversed_variance_ratio:.4f} w odwróconym porządku)")

# Indeks, dla którego suma wyjaśnionej wariancji jest równa lub większa niż 0.95
index_to_retain = np.argmax(cumulative_variance_ratios >= 0.95) + 1
print(f"\nLiczba komponentów do zachowania, aby zachować co najmniej 95% wariancji: {index_to_retain}")

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # Tworzymy subplot 3D
ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y, cmap='viridis')
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_zlabel('Petal length (cm)')
ax.set_title('Iris dataset - Sepal Length vs. Sepal Width vs. Petal Length')
plt.show()