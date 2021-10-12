import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

# Importing the data

X = pd.read_csv('cleaned_X.csv')
y = pd.read_csv('cleaned_y.csv')
y = y['Label_dark']

# Previewing the data to make sure it's the correct cleaned data

print(X)
print(y)

# Training and testing data split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.25, random_state = 1)

rf = RandomForestClassifier(random_state = 1)
print('Parameters currently in use:\n')
pprint(rf.get_params())
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# The randomized search for the best parameters

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

# What the randomized search found...

print('\n\n')
pprint(rf_random.best_params_)

# Evaluating best parameter search vs the base model

def evaluate(model, test_features, test_labels):
  predictions = model.predict(test_features)
  errors = abs(predictions - test_labels)
  accuracy = metrics.accuracy_score(test_labels, predictions)
  print('\n\n')
  print('Model Performance')
  print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
  print('Accuracy = {:0.2f}%.'.format(accuracy))
  
  return accuracy
base_model = RandomForestClassifier(n_estimators = 10, random_state = 1)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

# Showing the percentage of model improvement

print()
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))