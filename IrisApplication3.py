# Application 3 : Iris Dataset with Decision Tree (Supervised Machine Learning) 

# • In this application we remove one entry from each label of iris dataset and train with the remaining entries. 
# • And we apply predictions based on Decision tree with that removed entries Consider below characteristics of Machine Learning Application : 

# Classifier : Decision Tree 
# DataSet : Iris Dataset  
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length    
# Labels : Versicolor, Setosa , Virginica  
# Training Dataset : 147 Entries   
# Testing Dataset : 3 Entries 

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

print("Feature names of iris data set")
print(iris.feature_names)

print("Target names of iris data set")
print(iris.target_names)

test_index = [1,51,101]

train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis = 0)

test_target = iris.target[test_index]
test_data = iris.data[test_index]

classifier = tree.DecisionTreeClassifier()

classifier.fit(train_data,train_target)

print("Values that we removed from testing")
print(test_target)

print("Result of testing")
print(classifier.predict(test_data))