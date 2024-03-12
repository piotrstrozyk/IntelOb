import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=278795)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych
df = pd.read_csv("iris.csv")

# Podział danych na zbiór treningowy i testowy
all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

# Definicja wartości k dla k-NN
k_values = [3, 5, 11]

# Klasyfikacja k-NN dla różnych wartości k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_inputs, train_classes)
    predicted_classes = knn.predict(test_inputs)
    
    accuracy = accuracy_score(test_classes, predicted_classes)
    confusion = confusion_matrix(test_classes, predicted_classes)
    
    print(f"k-NN, k={k}")
    print("Procentowa dokładność:", accuracy)
    print("Macierz błędu:")
    print(confusion)
    print()

# Klasyfikacja Naive Bayes
nb = GaussianNB()
nb.fit(train_inputs, train_classes)
predicted_classes = nb.predict(test_inputs)

accuracy = accuracy_score(test_classes, predicted_classes)
confusion = confusion_matrix(test_classes, predicted_classes)

print("Naive Bayes")
print("Procentowa dokładność:", accuracy)
print("Macierz błędu:")
print(confusion)

# Wyświetlenie macierzy błędów dla klasyfikatora Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=df['variety'].unique(), yticklabels=df['variety'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Naive Bayes')
plt.show()