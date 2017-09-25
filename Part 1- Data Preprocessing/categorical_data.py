# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# creating the matrix of features 
X = dataset.iloc[:, :-1].values # normal slicing doesnt work with pandas 
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer # press ctrl+I to get more info on the imported class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) # fittingth the imputer object with the data
X[:, 1:3] = imputer.transform(X[:, 1:3]) # replacing the data with transform method 

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0]) # dummy variable creation # specify whick column to encode here it is [0]
X = onehotencoder.fit_transform(X).toarray() # fit and transform entire data, since the colum to be changed is pre-specified
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder() 
y = labelencoder_y.fit_transform(y) 

labelencoder_X.fit_transform(y)
