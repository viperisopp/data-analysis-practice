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

df = pd.read_csv("csves/red_wine.csv")


y = df["quality"].to_numpy()
x = df.drop("quality",axis=1)
x = x.to_numpy()
x = np.reshape(x,(-1,11))


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
plt.title("rated quality of wine vs predicted quality of wine")
plt.ylabel("quality of wine")
plt.legend(["actual","predicted"])

plt.show()