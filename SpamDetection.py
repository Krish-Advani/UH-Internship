from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np



#Predicition Model
data = pd.read_csv("spam.csv")
data=data.drop(['Email No.'],axis=1)

y = data["Prediction"].values
X = data[data.columns[:-1]].values
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X, y)

prompt = input("Type the prompt: ")
prompt=prompt.lower().split(" ")
X_new = []
for i in range(len(data.columns[:-1])):
	X_new.append(0)
	for j in prompt:
		if j==data.columns[i]:
			X_new[i]+=1
X_new=[X_new]
y_pred = knn.predict(X_new)
print("Predictions: {}".format(y_pred))



#Error Calculation 
X = data.drop("Prediction", axis=1).values
y = data["Prediction"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,y_train)

print("Accuracy: "+str(knn.score(X_test, y_test)))



# Graph Neighbors to Accuracy
neighbors = np.arange(1, 100)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
	knn = KNeighborsClassifier(n_neighbors=neighbor)
	knn.fit(X_train, y_train)
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)

plt.title("Accuracy vs. Number of Neighbors")
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

plt.show()