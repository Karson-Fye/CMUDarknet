import pandas as pd
import numpy as np
# from imblearn.over_sampling import SMOTE

def initData():
  df = pd.read_csv('../data/Darknet.CSV')
  if df.empty:
    print("Empty")
  return df

def cleanData(df):
  df['Label1'] = df['Label1'].str.lower()
  df.Label1.unique()

  threshold = 1
  for col in df.columns:
    if len(df[col].unique()) <= threshold:
      df.drop([col], axis=1, inplace=True)
  # print(df.shape) 

  df.drop_duplicates(inplace=True)
  # print(df.shape)

  df = df.replace([np.inf, -np.inf], np.nan)
  df = df.dropna()
  # print(df.shape)

  return df

def sepXY(df):
  df['Label_dark'] = df['Label'].apply(lambda x: 1 if x == 'VPN' or 
                                     x == 'Tor' else 0)

  X = df.select_dtypes(exclude=object)
  X = X.drop('Label_dark',axis=1)

  y = df[['Label_dark']]

  return (X,y)

def saveX(X):
  X.to_csv(r'../data/cleaned_X.csv', index=False)

def saveY(y):
  y.to_csv(r'../data/cleaned_y.csv', index=False)

def main():
  df = initData()
  df = cleanData(df)
  (X,y) = sepXY(df)
  saveX(X)
  saveY(y)

main()