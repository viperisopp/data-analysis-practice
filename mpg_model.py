import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from datetime import datetime
import random
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


df = pd.read_csv("csves/auto_mpg.csv")

df["horsepower"] = df["horsepower"].apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=0)

y = df["mpg"].to_numpy()
x = df[["weight","horsepower","cylinders"]].to_numpy()
x = np.reshape(x,(-1,3))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = LinearRegression()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)

print(model.coef_,model.intercept_)
print(mean_squared_error(y_test,y_predict))
print(r2_score(y_test,y_predict))

index1 = [n/2 for n in range(1,11)]
index2 = [n+0.1 for n in index1]

plt.bar(index1,y_test[:10],color="blue",width=0.1)
plt.bar(index2,y_predict[:10],color="orange",width=0.1)
plt.title("tested miles per gallon vs predicted miles per gallon")
plt.ylabel("mies per gallon")
plt.legend(["actual","predicted"])

plt.show()
