import numpy as np


class LinearRegression:
    def __int__(self):
        self.theta = None

    def fit(self, input_datax, datay, learning_rate, lamda):
        sample_num, character_num = input_datax.shape
        datax = np.c_[input_datax, np.ones(sample_num)]
        self.theta = np.zeros([character_num + 1, 1])
        maxstep = int(1e8)
        last_better = 0
        last_theta = int(1e8)
        stoplearn_value = 1e-8
        stoplearn_count = 50
        for step in range(0, maxstep):
            predict = datax.dot(self.theta)
            jtheta = sum((predict - datay) ** 2) / (2 * sample_num)
            self.theta = self.theta - learning_rate * (lamda * self.theta + (datax.T.dot(predict - datay)) / sample_num)
            if jtheta < last_theta - stoplearn_value:
                last_theta = jtheta
                last_better = step
            elif step - last_better > stoplearn_count:
                break;
            print("步数:%s--theta:%.6f" % (step, jtheta))

    def predict(self, input_datax):
        sample_num = input_datax.shape[0]
        datax = np.c_[input_datax, np.ones(sample_num)]
        return datax.dot(self.theta)
