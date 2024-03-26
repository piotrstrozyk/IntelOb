import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('diabetes.csv')

# Podział danych na cechy (X) i etykiety klas (y)
X = data.drop('class', axis=1)
y = data['class']

# Podział danych na zbiór treningowy i testowy
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# we fit the train data
scaler.fit(train_data)

# scaling the train data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


#1: relu, gorzej

# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(6,3), activation = 'relu', max_iter=500)

# mlp.fit(train_data, train_labels)

# from sklearn.metrics import accuracy_score, confusion_matrix

# #Dokładność na zbiorze testowym
# predictions_test = mlp.predict(test_data)
# print(accuracy_score(predictions_test, test_labels))

# conf_matrix = confusion_matrix(test_labels, predictions_test)
# print("Macierz błędu:")
# print(conf_matrix)

#2: tanh, gorzej

# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(6,3), activation = 'tanh', max_iter=500)

# mlp.fit(train_data, train_labels)

# from sklearn.metrics import accuracy_score, confusion_matrix

# #Dokładność na zbiorze testowym
# predictions_test = mlp.predict(test_data)
# print(accuracy_score(predictions_test, test_labels))

# conf_matrix = confusion_matrix(test_labels, predictions_test)
# print("Macierz błędu:")
# print(conf_matrix)

#3: logistic, gorzej

# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(6,3), activation = 'logistic', max_iter=500)

# mlp.fit(train_data, train_labels)

# from sklearn.metrics import accuracy_score, confusion_matrix

# #Dokładność na zbiorze testowym
# predictions_test = mlp.predict(test_data)
# print(accuracy_score(predictions_test, test_labels))

# conf_matrix = confusion_matrix(test_labels, predictions_test)
# print("Macierz błędu:")
# print(conf_matrix)

#4: identity, gorzej

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(6,3), activation = 'identity', max_iter=500)

mlp.fit(train_data, train_labels)

from sklearn.metrics import accuracy_score, confusion_matrix

#Dokładność na zbiorze testowym
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))

conf_matrix = confusion_matrix(test_labels, predictions_test)
print("Macierz błędu:")
print(conf_matrix)

#0.7402597402597403
# Macierz błędu:
# [[122  29]
#  [ 31  49]]
# FP (False Positive) 29 przypadków, gdzie model błędnie zaklasyfikował osoby bez cukrzycy jako osoby z cukrzycą.
# FN (False Negative) 31 przypadków, gdzie model błędnie zaklasyfikował osoby z cukrzycą jako osoby bez cukrzycy.
#Więcej FN niż FP, gorsze FN ze względu na ryzyko zdrowotne