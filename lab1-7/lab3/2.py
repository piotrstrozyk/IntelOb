import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")


all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)

tree_text = export_text(dtc, feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
print(tree_text)



plt.figure(figsize=(12, 8))
plot_tree(dtc, feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], class_names=df['variety'].unique(), filled=True)
plt.show()

accuracy = dtc.score(test_inputs, test_classes)

print("Dokładność klasyfikatora: {:.2f}%".format(accuracy * 100))
#95.56%

predicted_classes = dtc.predict(test_inputs)

conf_matrix = confusion_matrix(test_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=df['variety'].unique(), yticklabels=df['variety'].unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()