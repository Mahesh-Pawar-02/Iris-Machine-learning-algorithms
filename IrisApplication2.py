# Application 2 : Iris Dataset (Supervised Machine Learning.)

# In this application we are using Iris data set which contains information about the flowers under Iris family.
# This data set contains 4 Features as Sepal length, Sepal width, petal length and petal width.
# This data set contains 150 records.
# Consider below characteristics of Machine Learning Application : 

# Classifier : Decision Tree   
# DataSet : Iris Dataset    
# Features : Sepal Width, Sepal Length, Petal Width, Petal Length    
# Labels : Versicolor, Setosa , Virginica   
# Training Dataset : 150 Entries 
# Testing Dataset : -- 

# Consider below application which loads Iris dataset and display all records and labels of that data set

from sklearn.datasets import load_iris

iris = load_iris()

print("Feature names of Iris data set")
print(iris.feature_names)

print("Target names of Iris Data set")
print(iris.target_names)

print("First 10 element from iris data set")

for i in range(len(iris.target)):
    print("ID : %d, Label : %s, Feature : %s"%(i,iris.data[i],iris.target[i]))