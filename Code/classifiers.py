import pandas as pd
import numpy as np
from scipy import rand
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

def trainTest():
  X = pd.read_csv('../data/cleaned_X.csv')
  y = pd.read_csv('../data/cleaned_y.csv')
  y = y['Label_dark'].values

  return train_test_split(X,y,test_size=0.2,random_state=1)

def logRegression(X_train,X_test,y_train,y_test):
  print("Logistical Regression")
  lr = LogisticRegression(random_state=0, solver='sag')
  lr.fit(X_train, y_train)
  lr_pred = lr.predict(X_test)
  # accuracy.append(metrics.accuracy_score(y_test, lr_pred))
  # tests.append('Logistical Regression')
  printResults(lr,X_train,y_train,y_test,lr_pred)

def randomForest(X_train,X_test,y_train,y_test):
  print("Random Forest")
  rf = RandomForestClassifier(random_state=0)
  rf.fit(X_train , y_train)
  rf_pred = rf.predict(X_test)  
  # accuracy.append(metrics.accuracy_score(y_test, rf_pred))
  # tests.append('Random Forest')
  printResults(rf,X_train,y_train,y_test,rf_pred)

def gradientBoosting(X_train,X_test,y_train,y_test):
  print("Gradient Boosting")
  gb = GradientBoostingClassifier(random_state=0)
  gb.fit(X_train , y_train)
  gb_pred = gb.predict(X_test)
  # accuracy.append(metrics.accuracy_score(y_test, gb_pred))
  # tests.append('Gradient Boosting')
  printResults(gb,X_train,y_train,y_test,gb_pred)

def adaBoost(X_train,X_test,y_train,y_test):
  print("ADA Boost")
  ada = AdaBoostClassifier(random_state=0)
  ada.fit(X_train , y_train)
  ada_pred = ada.predict(X_test)
  # accuracy.append(metrics.accuracy_score(y_test, ada_pred))
  # tests.append('Ada Boost')
  printResults(ada,X_train,y_train,y_test,ada_pred)

def knn(X_train,X_test,y_train,y_test):
  print("K-Nearest Neighbors")
  knn = KNeighborsClassifier()
  knn.fit(X_train, y_train)
  knn_pred = knn.predict(X_test)
  # accuracy.append(metrics.accuracy_score(y_test, knn_pred))
  # tests.append('K-Nearest Neighbors')
  printResults(knn,X_train,y_train,y_test,knn_pred)

def sdg(X_train,X_test,y_train,y_test):
  print("SDG")
  sdg = SGDClassifier(random_state=0)
  sdg.fit(X_train, y_train)
  sdg_pred = sdg.predict(X_test)
  # accuracy.append(metrics.accuracy_score(y_test, sdg_pred))
  # tests.append('SDG Classifier')
  printResults(sdg,X_train,y_train,y_test,sdg_pred)

def printResults(classifier,X_train,y_train,y_test,predictions):
  print('\nAccuracy of training data', classifier.score(X_train, y_train))
  print('Accuracy of testing data ', metrics.accuracy_score(y_test, predictions), end='\n\n')
  print(metrics.confusion_matrix(y_test, predictions), end='\n\n')
  print(metrics.classification_report(y_test, predictions))

def main():
  X_train,X_test,y_train,y_test = trainTest()

  logRegression(X_train,X_test,y_train,y_test)
  randomForest(X_train,X_test,y_train,y_test)
  gradientBoosting(X_train,X_test,y_train,y_test)
  adaBoost(X_train,X_test,y_train,y_test)
  knn(X_train,X_test,y_train,y_test)
  sdg(X_train,X_test,y_train,y_test)

main()