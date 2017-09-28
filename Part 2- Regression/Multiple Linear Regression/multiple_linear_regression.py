# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray() # converting the object to array 

# Avoiding the Dummy Variable Trap-- removign redundant dependencies
## Python library for linear regression takes care of this, but it is good to be sure to do it manually
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# Not needed since the library will take care of this 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


########## Backward elimination ####################
# Building the optimal model using Backward Elimination 
import statsmodels.formula.api as sm  # this library does not inclue the intercept part. 
# So add a column of '1' for the intercept in the beginning 
X = np.append(arr= np.ones((50,1)).astype(int),values=X,axis=1) # First column is '1's and the rest is the matrix of data

# Selecting the data 
X_opt = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
# To look for the p-value and other statistical values 
regressor_ols.summary()
# highest p-value = x2 (index 2)

### second iteration 
X_opt = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
# To look for the p-value and other statistical values 
regressor_ols.summary()
# highest p-value = x1 (index 1)

### third iteration 
X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
# To look for the p-value and other statistical values 
regressor_ols.summary()
# highest p-value = x1 (index 1)

### 4rth iteration 
X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
# To look for the p-value and other statistical values 
regressor_ols.summary()
# highest p-value = x1 (index 1)

### 5th iteration 
X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog=y,exog=X_opt).fit()
# To look for the p-value and other statistical values 
regressor_ols.summary()

######  NOTE:
###### This is not the only criteria. There are several criteria to be checked for model performances. 