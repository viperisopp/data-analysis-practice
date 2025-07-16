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

df = pd.read_csv("tv_advertising.csv")

df = df.drop(columns=["index"])

tv = df["TV"].to_numpy()
radio = df["radio"].to_numpy()
newspaper = df["newspaper"].to_numpy()

adverts_combined = df[["TV","radio","newspaper"]].to_numpy()
y = df["sales"].to_numpy()


tv_train,tv_test,y_train_tv,y_test_tv = train_test_split(tv,y,test_size=0.2)
radio_train,radio_test,y_train_radio,y_test_radio = train_test_split(radio,y,test_size=0.2)
newspaper_train,newspaper_test,y_train_newspaper,y_test_newspaper = train_test_split(newspaper,y,test_size=0.2)
adverts_combined_train,adverts_combined_test,y_train_adverts_combined,y_test_adverts_combined = train_test_split(adverts_combined,y,test_size=0.2)


adverts_combined_model = LinearRegression()
adverts_combined_train = adverts_combined_train.reshape(-1,3)
adverts_combined_test = adverts_combined_test.reshape(-1,3)
adverts_combined_model = adverts_combined_model.fit(adverts_combined_train,y_train_adverts_combined)
adverts_combined_predict = adverts_combined_model.predict(adverts_combined_test)

tv_model = LinearRegression()
tv_train = tv_train.reshape(-1,1)
tv_test = tv_test.reshape(-1,1)
tv_model.fit(tv_train,y_train_tv)
tv_predict = tv_model.predict(tv_test)

radio_model = LinearRegression()
radio_train = radio_train.reshape(-1,1)
radio_test = radio_test.reshape(-1,1)
radio_model.fit(radio_train,y_train_radio)
radio_predict = radio_model.predict(radio_test)

newspaper_model = LinearRegression()
newspaper_train = newspaper_train.reshape(-1,1)
newspaper_test = newspaper_test.reshape(-1,1)
newspaper_model.fit(newspaper_train,y_train_newspaper)
newspaper_predict = newspaper_model.predict(newspaper_test)

plt.subplot(1,4,1)
plt.scatter(tv_train,y_train_tv,color="red")
plt.scatter(tv_test,y_test_tv,color="green")
plt.plot(tv_test,tv_predict,color="blue")
plt.title("tv")

plt.subplot(1,4,2)
plt.scatter(radio_train,y_train_radio,color="red")
plt.scatter(radio_test,y_test_radio,color="green")
plt.plot(radio_test,radio_predict,color="blue")
plt.title("radio")

plt.subplot(1,4,3)
plt.scatter(newspaper_train,y_train_newspaper,color="red")
plt.scatter(newspaper_test,y_test_newspaper,color="green")
plt.plot(newspaper_test,newspaper_predict,color="blue")
plt.title("newspaper")

index1 = [n/2 for n in range(1,11)]
index2 = [n+0.1 for n in index1]

plt.subplot(1,4,4)
plt.bar(index1,y_test_adverts_combined[:10],color="blue",width=0.1)
plt.bar(index2,adverts_combined_predict[:10],color="orange",width=0.1)
plt.legend(["actual","predicted"])
plt.title("actual sales vs predicted sales for test dataset")


plt.suptitle("train and test data")
plt.show()