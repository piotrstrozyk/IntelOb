import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Wczytanie danych
df = pd.read_csv("iris.csv")

# Podział danych na zbiór treningowy i testowy
all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

# Inicjalizacja klasyfikatorów
classifiers = {
    "Decision Tree": DecisionTreeClassifier(),
    "3-NN": KNeighborsClassifier(n_neighbors=3),
    "5-NN": KNeighborsClassifier(n_neighbors=5),
    "11-NN": KNeighborsClassifier(n_neighbors=11),
    "Naive Bayes": GaussianNB()
}

# Trenowanie i ocena każdego klasyfikatora
accuracy_results = {}
for name, clf in classifiers.items():
    clf.fit(train_inputs, train_classes)
    predicted_classes = clf.predict(test_inputs)
    accuracy = accuracy_score(test_classes, predicted_classes)
    accuracy_results[name] = accuracy

# Wyświetlenie wyników
print("Dokładność klasyfikatorów:")
for name, accuracy in accuracy_results.items():
    print(f"{name}: {accuracy}")

# Znalezienie klasyfikatora o najwyższej dokładności
best_classifier = max(accuracy_results, key=accuracy_results.get)
print(f"\nNajlepszy klasyfikator pod względem dokładności: {best_classifier} ({accuracy_results[best_classifier]})")
