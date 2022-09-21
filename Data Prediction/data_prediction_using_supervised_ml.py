# Task 1: Predict the percentage of a student based on the no. of study hours.

# Commented out IPython magic to ensure Python compatibility.
#Step 1: Importing Python libraries

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
# %matplotlib inline

#Step 2 : Importing data-set 

data=pd.read_csv(r'student_scores.csv',sep=',')

print("Importing data Successfully")

print("we print first 10 data of data-set ")
data.head(10)

print("dataset imported correctly")

# Step 4 : We Plot the Graph ,for Analysis of Dataset 

data.plot(x='Hours',y='Scores',style='1')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

data.plot.pie(x='Hours',y='Scores')

data.plot.scatter(x='Hours',y='Scores')

data.plot.bar(x='Hours',y='Scores')

# As Study Hours increases Score also increases that indicates correct data.
# Step 5 : We have prepared the data for our model
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
# print(X)

# Step 6 : we divide the data for training & testing of model

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)

# Training the Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 

regressor.fit(X_train, y_train) 
print("Training completed successfully.")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='orange');
plt.show()

# Step 7 : We test the model .
print(X_test) 
print("Predection of Score")
y_pred = regressor.predict(X_test)
print(y_pred)

# Step 8 : We Check the accuracy of our model
def new_func():
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  

    df

new_func()

# Step 9 : let's predict with custom input
hours = [[9.25]]
pred = regressor.predict(hours)
print(pred)

# Evaluating the model
from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))