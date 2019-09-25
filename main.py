import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import matplotlib.pyplot as plt
from sklearn import model_selection

def importdata(): 
    balance_data = pd.read_csv( 
'https://archive.ics.uci.edu/ml/machine-learning-'+
'databases/balance-scale/balance-scale.data', 
    sep= ',', header = None) 
    
    return balance_data 

def splitdataset(balance_data): 
    X = balance_data.values[:, 1:5] 
    Y = balance_data.values[:, 0] 
    # Seperating the target variable 
   
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y, X_train, X_test, y_train, y_test 

def train_using_gini(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 


def prediction(X_test, clf_object):

    y_pred = clf_object.predict(X_test)
    return y_pred

def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ")

    print(confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
def boxplot(clf, X, Y):
    kfold = model_selection.KFold(n_splits=10)
    results = model_selection.cross_val_score(clf, X, Y, cv=kfold, scoring='accuracy')

    print('Mean accuracy:', results.mean())
    print('St Dev accuracy:', results.std())
    fig = plt.figure()
    fig.suptitle('gini')
    plt.boxplot(results)
    plt.show()

def main(): 
      
    data = importdata()
    # Building Phase 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, X_test, y_train) 

    # Prediction using entropy
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 

    boxplot(clf_gini, X, Y)

if __name__=="__main__":
    main()



