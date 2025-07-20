import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)

x = df[["age","sex","bmi"]].to_numpy()
y = diabetes.target
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
plt.legend(["actual","predicted"])

plt.show()