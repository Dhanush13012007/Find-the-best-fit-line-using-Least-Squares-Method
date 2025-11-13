# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: M.Dhanush
RegisterNumber: 25009955  
*/
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

print ("Dataset:\n",df.head())
df

x = df[["Hours_Studied"]]   # Independent variable
y = df["Marks_Scored"]      # Dependent variable

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42

)

model= LinearRegression()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

print("\nModel Parameteters:")
print("Intercept (b0):", model.intercept_)
print("slope (b1):", model.coef_[0])


print("\nEvaluation Metrics:")
print("Mean Squared Error:",mean_squared_error(y_test, y_pred))
print("R^2 Score:",r2_score(y_test, y_pred))

plt.figure(figsize=(8,6))
plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, model.predict(x), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

## Output:
![best fit line](sam.png)

<img width="687" height="545" alt="image" src="https://github.com/user-attachments/assets/56f814ab-6da7-400a-8001-131ad2de6cb6" />

hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")

<img width="1253" height="108" alt="Screenshot 2025-11-13 092713" src="https://github.com/user-attachments/assets/39f6c326-55c7-49cd-860b-bc23fe8dbecc" />


## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
