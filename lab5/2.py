#a) #W sekcji preprocessingu, dane są przygotowywane do użycia w modelu sieci neuronowej

#reshape: Obrazy z zestawu danych MNIST są 2D (28x28 pikseli), 
# ale sieci konwolucyjne wymagają obrazów 3D (wysokość x szerokość x kanały). 
# Ponieważ obrazy MNIST są w skali szarości, mamy tylko jeden kanał, 
# więc zmieniamy kształt obrazów z (28, 28) na (28, 28, 1). Dodatkowo, 
# dzielimy wartości pikseli przez 255, 
# aby przeskalować je do zakresu [0, 1], 
# co jest często lepsze dla uczenia sieci neuronowych.

#to_categorical: Etykiety klas w zestawie danych MNIST są liczbami całkowitymi od 0 do 9. 
# Funkcja to_categorical jest używana do przekształcenia tych etykiet w one-hot encoding, 
# co jest wymagane dla wieloklasowej klasyfikacji z użyciem funkcji straty categorical_crossentropy. 
# One-hot encoding to reprezentacja, 
# w której etykieta n jest przekształcana w wektor o długości 10 (dla MNIST) z 1 na 
# n-tej pozycji i 0 na pozostałych pozycjach.

#Funkcja np.argmax jest używana do odzyskania oryginalnych etykiet z one-hot encoding. 
# Zwraca indeks (czyli klasę) o najwyższej wartości dla każdego wektora w test_labels. 
# Te oryginalne etykiety są zapisywane do późniejszego użycia przy tworzeniu macierzy pomyłek.


#b) Każda warstwa przekształca dane wejściowe w dane wyjściowe, które są następnie przekazywane do następnej warstwy. Oto, jak to działa dla poszczególnych warstw w podanym modelu:

# Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)): 
# Ta warstwa konwolucyjna przyjmuje obrazy o kształcie (28, 28, 1) i przekształca je w 32 mapy cech, każda o kształcie (26, 26),
# używając filtrów o kształcie (3, 3). Funkcja aktywacji ReLU jest stosowana do każdej mapy cech.

# MaxPooling2D((2, 2)): Ta warstwa pooling przekształca 32 mapy cech o kształcie (26, 26) w 32 mapy cech o kształcie (13, 13) 
# poprzez zastosowanie operacji max pooling z oknem o kształcie (2, 2).

# Flatten(): Ta warstwa spłaszcza 32 mapy cech o kształcie (13, 13) do jednego wektora o długości 5408 (32 * 13 * 13).

# Dense(64, activation='relu'): Ta warstwa gęsta przekształca wektor o długości 5408 w wektor o długości 64, stosując transformację liniową 
# (mnożenie przez macierz wag i dodanie biasu) i funkcję aktywacji ReLU.

# Dense(10, activation='softmax'): Ta warstwa gęsta przekształca wektor o długości 64 w wektor o długości 10, 
# stosując transformację liniową i funkcję aktywacji softmax. Wynikowy wektor reprezentuje prawdopodobieństwa przynależności do każdej z 10 klas.

#c) 4 jest często mylone z 9 (18 razy), tak samo 7 z 9 (10 razy)

#d) Model jest nieco przeuczony (train accuracy jest wyższe niż validation accuracy)


#e) dodałem ModelCheckpoint