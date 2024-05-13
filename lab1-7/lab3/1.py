import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv")

#podzial na zbior testowy (30%) i treningowy (70%), ziarno losowosci = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=278795)

# train_inputs = train_set[:, 0:4]
# train_classes = train_set[:, 4]
# test_inputs = test_set[:, 0:4]
# test_classes = test_set[:, 4]

# print(test_set)
# print(test_set.shape[0])

def classify_iris(sl, sw, pl, pw):
    if pw < 1:
        return("Setosa")
    elif pw > 1.5 and pl > 4:
        return("Virginica")
    else:
        return("Versicolor")
    
good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]) == test_set[i, -1]:
        good_predictions += 1

        
print(good_predictions)
print(good_predictions/len*100, "%")

grouped = df.groupby('variety')

mean_by_species = grouped.mean()

print(mean_by_species)

sorted = sorted(train_set, key=lambda x: x[-1])
for row in sorted:
    print(row)

# 97.77777777777777 %