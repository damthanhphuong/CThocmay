from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
# Chiều cao - Biến giải thích X
x = np.array([[ 147, 150, 158, 160, 163,168, 170, 178, 180, 183]]).T
# Cân nặng, Biến mục tiêu - Y
y = np.array([[49, 50, 54, 56, 58,60, 72, 66, 67, 68 ]]).T
# Mô tả, vẽ dữ liệu
plt.plot(x, y, 'bo')
plt.axis([ 140,190,40, 80, ])
plt.xlabel('Chiều cao')
plt.ylabel('Cân nặng')
plt.show()
# Xây dựng ma trận dữ liệu mở rộng
one = np.ones((x.shape[0], 1))
Xbar = np.concatenate((one, x), axis = 1)

# Tính toán đường hồi quy
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('Tham số của đường thẳng là:', w)
# Viết phương trình đường thẳng
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(140, 180, 2)
y0 = w_0 + w_1*x0

# Vẽ đường hồi quy
plt.plot(x.T, y.T, 'bo')     # Dữ liệu
plt.plot(x0, y0)               # Đường hồi quy
plt.axis([ 140,190,40, 80, ])
plt.xlabel('A')
plt.ylabel('B')
plt.show()
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print (u'Nghiệm tìm được bằng scikit-learn  : ', regr.coef_)
print (u'Nghiệm tìm được từ phương trình (5): ', w.T)
yc = w_1*60 + w_0
#y2 = w_1*160 + w_0

print( u'Dự đoán giá của cá ngừ nặng 60 kg: %.2f (kg), dữ liệu thật: chưa có (kg)'  %(yc) )
#print( u'Dự đoán cân nặng của người cao 160 cm: %.2f (kg), dữ liệu thật: 56 (kg)'  %(y2) )
