from venv import create
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
import classifiers

def createGrid():
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  max_features = ['sqrt', 'log2']
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  min_samples_split = [2, 5, 10]
  min_samples_leaf = [1, 2, 4]
  bootstrap = [True, False]

  return {'n_estimators': n_estimators,
          'max_features': max_features,
          'max_depth': max_depth,
          'min_samples_split': min_samples_split,
          'min_samples_leaf': min_samples_leaf,
          'bootstrap': bootstrap}

def evaluateFeatures(model, test_features, test_labels):
  predictions = model.predict(test_features)
  errors = abs(predictions - test_labels)
  accuracy = metrics.accuracy_score(test_labels, predictions)

  print('\n\n')
  print('Model Performance')
  print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
  print('Accuracy = {:0.4f}%.'.format(accuracy))
  
  return accuracy

def hyperTuning(X_train,X_test,y_train,y_test,random_grid):
  rf = RandomForestClassifier(random_state = 1)
  pprint(rf.get_params())
  rf_random = RandomizedSearchCV(estimator = rf, 
                    param_distributions = random_grid,
                    n_iter = 300, cv = 3, verbose=2, 
                    random_state=42, n_jobs = -1)
  rf_random.fit(X_train, y_train)
  pprint(rf_random.best_params_)
  
  base_model = RandomForestClassifier(n_estimators = 10, random_state = 1)
  base_model.fit(X_train, y_train)
  base_accuracy = hyperTuning(base_model, X_test, y_test)

  best_random = rf_random.best_estimator_
  random_accuracy = hyperTuning(best_random, X_test, y_test)

  return (base_accuracy,random_accuracy)

def results(random_accuracy, base_accuracy):
  improvement = 100*(random_accuracy-base_accuracy)/base_accuracy
  print('Improvement of {:0.2f}%.'.format(improvement))

def main():
  X_train,X_test,y_train,y_test = classifiers.trainTest()

  random_grid = createGrid()
  (base,random) = hyperTuning(X_train,X_test,y_train,y_test,random_grid)
  results(base,random)


main()

'''
{'bootstrap': True,
 'max_depth': 20,
 'max_features': 'sqrt',
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 1000}

Model Performance
Average Error: 0.0199 degrees.
Accuracy = 98%

Model Performance
Average Error: 0.0162 degrees.
Accuracy = 98%

Improvement of 0.38%.
'''