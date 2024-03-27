#a) Jest używana do przeskalowania cech numerycznych w zestawie danych, tak 
# aby miały średnią wartość 0 i odchylenie standardowe 1.

#Oblicza średnią wartość (mean) i odchylenie standardowe (std) dla każdej cechy w zestawie danych.

#Dla każdej wartości cechy x w zestawie danych, 
# przekształca x na (x - mean) / std.

#To przekształcenie jest nazywane standardyzacją lub normalizacją Z-score. 
# Jest to przydatne w wielu algorytmach uczenia maszynowego, które zakładają, że wszystkie cechy są centrowane wokół zera i mają tę samą skalę.


######################################################################



#b) Kodowanie "one hot", znane również jako kodowanie jednego z n, 
# to proces przekształcania kategorii na formę, która może być dostarczona do 
# algorytmów uczenia maszynowego, aby poprawić dokładność predykcji. 
# Kategorie są zazwyczaj reprezentowane jako ciągi tekstowe, 
# które są trudne do analizy przez algorytmy uczenia maszynowego.

#Przy kodowaniu "one hot", dla każdej unikalnej kategorii w danych, 
# tworzymy nową kolumnę (lub wymiar) i używamy binarnej reprezentacji (0 lub 1), 
# aby wskazać obecność danej kategorii. Na przykład, jeśli mamy trzy kategorie: 
# "czerwony", "zielony" i "niebieski", po zastosowaniu kodowania "one hot" otrzymamy:

# czerwony -> [1, 0, 0]
# zielony  -> [0, 1, 0]
# niebieski -> [0, 0, 1]

#Tak więc, OneHotEncoder przekształca etykiety klas na binarne wektory, 
# gdzie każda klasa jest reprezentowana przez wektor, 
# który ma 1 na pozycji odpowiadającej tej klasie i 0 na pozycjach odpowiadających 
# innym klasom.


#################################################################


#c) X_train.shape[1] zwraca liczbę kolumn w X_train, co odpowiada liczbie cech. 
# Więc, jeśli X_train.shape[1] wynosi 4, to znaczy, że mamy 4 cechy wejściowe i warstwa wejściowa ma 4 neurony.

#y_encoded.shape[1] zwraca liczbę kolumn w y_encoded, co odpowiada liczbie klas. 
# Więc, jeśli y_encoded.shape[1] wynosi 3, to znaczy, że mamy 3 możliwe klasy wyjściowe i 
# warstwa wyjściowa ma 3 neurony.




######################################################################

#d) tanh: test accuracy: 97.78%
# relu: test accuracy: 100%  --> najlepsza
# sigmoid: test accuracy: 95.56%



######################################################################

#e)Optimizer (optymalizator): Określa algorytm optymalizacji, 
# który ma być używany do aktualizacji wag modelu podczas uczenia.:  
# 'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'

#Loss (strata): Funkcja straty, która jest minimalizowana podczas uczenia modelu.
# np. dla problemów klasyfikacji wieloklasowej często stosuje się 'categorical_crossentropy', 
# dla regresji można użyć 'mean_squared_error' lub 'mean_absolute_error'.

#Metrics (metryki): Określa metryki, 
# które mają być obliczane podczas uczenia modelu. Oprócz 'accuracy', można użyć innych metryk, takich jak 
# 'precision', 'recall', 'f1-score' itp.

#Szybkość uczenia się jest jednym z kluczowych hiperparametrów w procesie uczenia się sieci neuronowych,
# który określa jak duże kroki są wykonywane podczas aktualizacji wag modelu podczas procesu uczenia
#custom_optimizer = Adam(learning_rate=0.001)


######################################################################

#f) Mniejszy rozmiar partii może prowadzić do bardziej gwałtownych fluktuacji w 
# krzywych uczenia się, ponieważ sieć jest aktualizowana częściej z mniejszymi porcjami danych.
# Z drugiej strony, większy rozmiar partii może prowadzić do bardziej gładkich krzywych uczenia się,
# ponieważ sieć jest aktualizowana rzadziej z większymi porcjami danych.

######################################################################  


#g) Dokładność na zestawie treningowym i walidacyjnym jest podobna i wysoka, 
# model jest prawdopodobnie dobrze dopasowany.
#Najlepsza epoka to taka, w której dokładność na zestawie walidacyjnym była najwyższa. np epoka 93/100

######################################################################

#h) Ten kod jest używany do przetwarzania, 
# trenowania i oceny modelu sieci neuronowej na zbiorze danych Iris
# import numpy as np 
# from sklearn.datasets import load_iris 
# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import StandardScaler, OneHotEncoder 
# from tensorflow.keras.models import load_model 
 
# # Load the iris dataset 
# iris = load_iris() 
# X = iris.data 
# y = iris.target 
 
# # Preprocess the data 
# # Scale the features 
# scaler = StandardScaler() 
# X_scaled = scaler.fit_transform(X) 
 
# # Encode the labels 
# encoder = OneHotEncoder(sparse=False) 
# y_encoded = encoder.fit_transform(y.reshape(-1, 1)) 
 
# # Split the dataset into training and test sets 
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, 
# random_state=42) 
 
# # Load the pre-trained model 
# model = load_model('iris_model.h5') 
 
# # Continue training the model for 10 more epochs 
# model.fit(X_train, y_train, epochs=10) 
 
# # Save the updated model 
# model.save('updated_iris_model.h5') 
 
# # Evaluate the updated model on the test set 
# test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) 
# print(f"Test Accuracy: {test_accuracy*100:.2f}%")