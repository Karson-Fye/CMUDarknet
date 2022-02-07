import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr

def initData():
  X = pd.read_csv('../data/cleaned_X.csv')
  y = pd.read_csv('../data/cleaned_y.csv')
  if X.empty or y.empty:
    print("Empty")
  return (X,y)

def chiSquared(X,y):
  chi_X=X
  chi_Y=y

  scaler = MinMaxScaler()
  chi_X = pd.DataFrame( 
      scaler.fit_transform(chi_X), 
      columns=chi_X.columns 
  )

  chi_scores = chi2(chi_X, chi_Y)
  p_vals = pd.Series(chi_scores[1],index = chi_X.columns)
  p_vals.sort_values(ascending=True, inplace=True)

  # print("\t X^2 p-values")
  # print(p_vals)

  return p_vals

def pCorrelation(X,y):
  pears_X = X.copy(deep=True)
  pears_Y = y.copy(deep=True)

  # Calculate r-value for each feature
  # Store results in a dictionary with the label and r-value
  pears_dict = {}
  for i in pears_X.columns:
    corr, _ = pearsonr(pears_X[i], pears_Y)
    pears_dict[pears_X[i].name] = corr

  pears_X = pd.DataFrame.from_dict(pears_dict, orient='index')
  pears_X.columns = ['r-value']

  pears_X['r-value'] = pears_X['r-value'].abs()
  pears_X.sort_values('r-value', inplace=True, ascending=False)
  r_vals = pears_X.squeeze()

  # print('\t Pearson r-Values')
  # print(r_vals)

  return r_vals

def topFeatures(X,p_vals,r_vals,topNum):
  top_features = [x for x in r_vals.index[:topNum] if x in p_vals.index[:topNum]]
  return X[top_features]


def main():
  (X,y) = initData()
  p_vals = chiSquared(X,y)
  r_vals = pCorrelation(X,y)
  selection = topFeatures(X,p_vals,r_vals,40)
  print(selection.columns)

main()