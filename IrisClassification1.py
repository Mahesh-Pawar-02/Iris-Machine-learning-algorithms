from sklearn import tree
from sklearn.datasets import load_iris

def main():
    print("--------Iris flower Classification case study-----------")

    iris = load_iris()

    print(iris)
    # print(type(data))

    Features = iris.data
    Labels = iris.target

    print("Feature are : ")
    print(Features)

    print("Labels are")
    print(Labels)
    
if __name__ == "__main__":
    main()
