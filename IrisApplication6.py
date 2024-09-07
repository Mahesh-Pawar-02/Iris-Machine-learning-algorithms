from sklearn import tree
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def euc(a,b):
    return distance.euclidean(a,b)

class Marvellous_KNN():
    def fit(self, TrainingData, TrainingTarget):
        self.TrainigData = TrainingData
        self.TrainigTarget = TrainingTarget

    def predict(self, TestData):
        predictions = []
        for row in TestData:
            lebel = self.closest(row)
            predictions.append(lebel)
        return predictions
    
    def closest(self,row):
        bestdistance = euc(row, self.TrainigData[0])
        bestindex = 0
        for i in range(1, len(self.TrainigData)):
            dist = euc(row, self.TrainigData[i])
            if dist < bestdistance:
                bestdistance = dist
                bestindex = i
        return self.TrainigTarget[bestindex]
    
def Marvellous_KNeighbhor():
    border = "-"*50

    iris = load_iris()

    data =iris.data
    target = iris.target


    print(border)
    print("Actual Data Set")
    print(border)

    for i in range(len(iris.target)):
        print("ID : %d, Label : %s, Feature : %s"%(i,iris.data[i],iris.target[i]))

    print("Size of Actul Data Set %d"%(i+1))

    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = 0.5)

    print(border)
    print("Training data sets")
    print(border)

    for i in range(len(data_train)):
        print("ID : %d, Label : %s, Feature : %s"%(i, data_train[i],target_train[i]))
    print("Size of Training data set %d"%(i+1))

    print("border")
    print("Test data set")
    print(border)

    for i in range(len(data_test)):
        print("ID : %d, Label : %s, Feature : %s" %(i,data_test[i], target_test[i]))
    print("Size of test data set %d"%(i+1))

    classifier = Marvellous_KNN()
    classifier.fit(data_train,target_train)
    predictions = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,predictions)
    return Accuracy

def main():
    Accuracy = Marvellous_KNeighbhor()
    print("Accuracy of classification algorithm with K Neighbor classifier is",Accuracy*100,"%")

if __name__ == "__main__":
    main()