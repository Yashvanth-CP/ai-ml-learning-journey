import numpy as np

w = np.array([1,2,3,4,5,6,7])
b =100
d = np.array([1, 22,33,34,44,55,55])
x = np.array([1, 2, 3, 4, 5, 6, 7])
# f = np.dot(w, x) + b
# print(f)
f = 0
for j in range (0,7):
    f =f + w[j] * x[j]

f = f + b
print(f)

# # gradient descent
# for j in range(0,6):
#     w[j] = w[j] - 0.1 * d[j]

# print(w)
learning_rate = 0.1
w = w - learning_rate * d
print(w)