# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
import os;
print(os.getcwd()) # Prints the working directory
dataset = pd.read_csv('C:\\Users\\PONNAS-CONT\\Desktop\\Machine Learning A-Z Template Folder\\Part 1 - Data Preprocessing\\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# FEATURE SELECTION--- Most of the ML packages support inbuilt feature scaling
## 1. This is done because most of the MLs use Eucledian distance for finding space
##     B/w the coordinates. So it may not capture the true relationship. So, the variables 
##     should be brought to the same scale
## 1.a. Even if not based on euclidean(decision trees), this enables faster convergence 
## 2. STANDARDISATION = X-mean/SD       Columns have zero mean and Unit variance
## 3. Normalisation = X-min/max-min     Values become(0 to 1). Good treament if putliers are present 
## 4. Dummy variables neeed not be scaled, thereby enabling easy interpretation

# Feature Scaling
# 1. This is dome after spltting the data
# 2. The mean and Variance of onyl the training data is stored and test data is transformed depending on that
##   This limits the exposure of model to qualities of test data
from sklearn.preprocessing import StandardScaler ## Standardisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) ## fitted with mean and var of train data only
X_test = sc_X.transform(X_test) ## transformed using fit from mean and var of train data
