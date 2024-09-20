import numpy as np
import linearregression
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 8000).reshape(-1, 1)
y = np.sin(x)

datax = np.ones((x.shape[0], 6))
for i in range(1, 6):
    datax[:, i] = x[:, 0] ** i

mean = np.mean(datax, axis=0)
std = np.std(datax, axis=0)

if np.any(std == 0):
    std[std == 0] = 1e-9

datax = (datax - mean) / std

model = linearregression.LinearRegression()

model.fit(datax, y, learning_rate=0.3, lamda=0.00005)

predict_y = model.predict(datax)

plt.figure()
plt.plot(x, y, "r-", lw=1.6, label='sin(x)')
plt.plot(x, predict_y, 'g-', lw=1.6, label='predict')
plt.legend(loc='best')
plt.grid(True)
plt.show()
