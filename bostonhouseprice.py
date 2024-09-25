import numpy as np
import pandas as pd
import linearregression
import matplotlib.pyplot as plt

#数据准备
data = pd.read_csv("housing-data.csv", header=None)
row_num = data.shape[0]
col_num = len(data.iloc[0, 0].split())
datax = np.empty([row_num, col_num - 1])
datay = np.empty([row_num, 1])
for i in range(0, row_num):
    num = data.iloc[i, 0].split()
    datax[i] = np.array(num[0:-1])
    datay[i] = np.array(num[-1])

#特征处理
maxdatax = datax.max(axis=0)
mindatax = datax.min(axis=0)
datax = (datax - mindatax) / (maxdatax - mindatax)

#调用算法
model = linearregression.LinearRegression()
model.fit(datax,datay,learning_rate=0.5,lamda=0.00003)
predict_data = model.predict(datax)

#画图
t = np.arange(len(predict_data))
plt.figure()
plt.scatter(t,datay,color ='g',label="real value",s=30)
plt.plot(t,predict_data,'r-',lw = 1.6 ,label = "predict value")
plt.legend(loc = 'best')
plt.title('Boston house price', fontsize=18)
plt.xlabel('Case ID', fontsize=15)
plt.ylabel('House price', fontsize=15)
plt.grid(True)
plt.show()
