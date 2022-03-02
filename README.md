### The concepts of Multivariate Regression and Gradient Boosting description: It descripes the analysis and gives intution for Multivariate Regression and Gradient Boosting


1.  **What is XGBoost?**  
    
XGBoost is an optimized Gradient Boosting Machine Learning library. It is usually written in C++, but has API in several other languages. The XGBoost algorithm is parallelizable i.e. it does parallelization within a single tree

2.  **Cons of using XGBoost:**
    
1)It is one of the most powerful algorithms with high performance. 2)It can harness all the processing power of modern multicore computers. 3)It is feasible to train on huge datasets. 4)Consistently outperform all single algorithm methods.

3.  **what is Extreme Gradient Boosting(XGBoost)?**
It is an open-source library that provides an efficient and effective implementation of the gradient. Although other open-source implementations of the approach existed before XGBoost, the release of XGBoost appeared to release the power of the technique and made the applied machine learning community take notice of gradient boosting more generally.

Gradient Boosting refers to a class of ensemble machine learning algorithms that can be used for classification or regression predictive modeling problems. Ensembles are constructed from decision tree models. Trees are added one at a time to the ensemble and fit to correct the prediction errors made by prior models. This is a type of ensemble machine learning model referred to as boosting. Models are fit using any arbitrary differentiable loss function and gradient descent optimization algorithm. This gives the technique its name, “gradient boosting,” as the loss gradient is minimized as the model is fit, much like a neural network.

4.  **What is multivariate regression?**
    

It is a supervised machine learning algorithm involving multiple data variables for analysis. Multivariate regression is an extension of multiple regression with one dependent variable and multiple independent variables Multivariate regression tries to find out a formula that can explain how factors in variables respond simultaneously to changes in others.

5.  **Where multivariate regression can be used??**
    
-Praneeta wants to estimate the price of a house. She will collect details such as the location of the house, number of bedrooms, size in square feet, amenities available, or not. Basis these details price of the house can be predicted and how each variables are interrelated.
-An agriculture scientist wants to predict the total crop yield expected for the summer. He collected details of the expected amount of rainfall, fertilizers to be used, and soil conditions. By building a Multivariate regression model scientists can predict his crop yield. With the crop yield, the scientist also tries to understand the relationship among the variables.

-If an organization wants to know how much it has to pay to a new hire, they will take into account many details such as education level, number of experience, job location, has niche skill or not. Basis this information salary of an employee can be predicted, how these variables help in estimating the salary. -Economists can use Multivariate regression to predict the GDP growth of a state or a country based on parameters like total amount spent by consumers, import expenditure, total gains from exports, total savings, etc.

-A company wants to predict the electricity bill of an apartment, the details needed here are the number of flats, the number of appliances in usage, the number of people at home, etc. With the help of these variables, the electricity bill can be predicted.

* * *

## CODE

```python
import numpy as np import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv("BostonHousingData.csv") 
X = dataset[["tract" , "longitude" , "latitude" , "crime" , "residential" , "industrial" , "nox" , "rooms" , "older" , "distance" , "highway" , "tax" , "ptratio" , "lstat"]].values  
y = dataset[["cmedv"]].values 
X.shape
y.shape
```
#### Output
```python
(506, 14)
(506, 1)
```

```python
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.linear_model import LinearRegression   `

### k-fold CV (using all the 14 variables) for 5 times

lm = LinearRegression()   `

### can tune other metrics, such as MSE

scores = cross_val_score(lm, X, y, scoring='neg_mean_squared_error', cv=5) 
scores  
```
#### Output
```python
array([-25.82783243, -27.23794898, -30.3510623 , -80.25664814,
       -31.40953455])
```

```python
### Xgboost
from xgboost import XGBRegressor

my_model = XGBRegressor()
# can tune other metrics, such as MSE
scores = cross_val_score(my_model, X, y, scoring='neg_mean_squared_error', cv=5)
scores
```

#### Output 
```python
array([ -6.78874155,  -9.79002285, -19.71282514, -41.26469772,
       -16.62709911])
```

```python
dataset = pd.read_csv("Cars.csv") 
 
X = dataset[["MPG" , "CYL" , "ENG"]].values 

y = dataset[["WGT"]].values 
```


### Multiple Linear Regression  
```python

### k-fold CV (using all the 3 variables) for 5 times

lm = LinearRegression()   `

### can tune other metrics, such as MSE

scores = cross_val_score(lm, X, y, scoring='neg_mean_squared_error', cv=5)
scores 
```
#### Output
```python
array([-168066.74726422,  -94144.34580807,  -78223.41445844,
        -48007.71189022, -116785.90395613])
```

```python
### XGBOOST 
from xgboost import XGBRegressor  
my_model = XGBRegressor()   `

### can tune other metrics, such as MSE

scores = cross_val_score(my_model, X, y, scoring='neg_mean_squared_error', cv=5) 
scores 
```
#### Output: 

```python
array([-186522.5951182 ,  -86182.18667086,  -82654.3741858 ,
        -34760.41135122,  -43382.35990583])
```
##### In this dataset the two models have different scores , sometimes the negative MSE for XGBRegressor is better than simple linear regression , and other times simple linear regression is better than XGBRegressor
