from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from  sklearn.model_selection import train_test_split

def main():
    print("--------Iris flower Classification case study-----------")

    iris = load_iris()

    print(iris)

    Features = iris.data
    Labels = iris.target

    data_train, data_test, targt_train, target_test = train_test_split(Features,Labels,test_size=0.5)
    
    obj = tree.DecisionTreeClassifier()

    obj = obj.fit(data_train,targt_train)

    output = obj.predict(data_test)

    Accuracy = accuracy_score(target_test,output)
    print("Acuracy is : ", Accuracy*100,"%")
    
if __name__ == "__main__":
    main()
