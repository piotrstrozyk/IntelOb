from sklearn.datasets import load_iris

iris = load_iris()

# a) splitting into train and test datasets

from sklearn.model_selection import train_test_split
datasets = train_test_split(iris.data, iris.target,
                            test_size=0.3)

train_data, test_data, train_labels, test_labels = datasets
print(iris)
# b)  etykiety klas są reprezentowane jako liczby całkowite od 0 do 2, które odpowiadają różnym gatunkom irysów:

    # 0 odpowiada gatunkowi Iris-setosa
    # 1 odpowiada gatunkowi Iris-versicolor
    # 2 odpowiada gatunkowi Iris-virginica

# c) scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# we fit the train data
scaler.fit(train_data)

# scaling the train data
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

#print(train_data[:3])

from sklearn.neural_network import MLPClassifier
# creating an classifier from the model:
# mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=2000)
# -> 0.9619047619047619
#    0.9555555555555556

#mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000)
# -> 0.9714285714285714
#    0.9777777777777777
#Winner

mlp = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=2000)
# ->  0.9714285714285714
#     0.9555555555555556

# let's fit the training data to our model
mlp.fit(train_data, train_labels)

from sklearn.metrics import accuracy_score

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_labels))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_labels))