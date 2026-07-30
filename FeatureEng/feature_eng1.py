# import numpy as np
# import matplotlib.pyplot as plt
# from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
# np.set_printoptions(precision=2) 

# # create target data
# x = np.arange(0, 20, 1)
# y = 1 + x**2
# X = x.reshape(-1, 1)

# model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")
# plt.plot(x,X@model_w + model_b, label="Predicted Value");  plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()

# # create target data
# x = np.arange(0, 20, 1)
# y = x**2

# # engineer features .
# X = np.c_[x, x**2, x**3]   #<-- added engineered feature
# X_features = ['x','x^2','x^3']

# fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X[:,i],y)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("y")
# plt.show()

# # create target data
# x = np.arange(0,20,1)
# X = np.c_[x, x**2, x**3]
# print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# # add mean_normalization 
# X = zscore_normalize_features(X)     
# print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# x = np.arange(0,20,1)
# y = x**2

# X = np.c_[x, x**2, x**3]
# X = zscore_normalize_features(X) 

# model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
# plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# x = np.arange(0,20,1)
# y = np.cos(x/2)

# X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
# X = zscore_normalize_features(X) 

# model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

# plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
# plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng

np.set_printoptions(precision=2)


x = np.arange(20)
y = 1 + x**2
X = x.reshape(-1, 1)

model_w, model_b = run_gradient_descent_feng(
    X, y,
    iterations=1000,
    alpha=1e-2
)

prediction = X @ model_w + model_b

plt.figure(figsize=(6,4))
plt.scatter(x, y, color='red', marker='x', label='Actual Value')
plt.plot(x, prediction, label='Predicted Value')
plt.title("Without Feature Engineering")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()




x = np.arange(20)
y = x**2

X_poly = np.c_[x, x**2, x**3]
feature_names = ['x', 'x²', 'x³']

fig, ax = plt.subplots(1, 3, figsize=(12,4), sharey=True)

for i in range(3):
    ax[i].scatter(X_poly[:, i], y)
    ax[i].set_xlabel(feature_names[i])
    ax[i].grid(True)

ax[0].set_ylabel("y")
plt.tight_layout()
plt.show()




x = np.arange(20)

X_poly = np.c_[x, x**2, x**3]

print("Peak-to-Peak range (Raw):")
print(np.ptp(X_poly, axis=0))

X_norm, mu, sigma = zscore_normalize_features(X_poly)

print("\nPeak-to-Peak range (Normalized):")
print(np.ptp(X_norm, axis=0))




x = np.arange(20)
y = x**2

X_poly = np.c_[x, x**2, x**3]
X_norm, mu, sigma = zscore_normalize_features(X_poly)

model_w, model_b = run_gradient_descent_feng(
    X_norm,
    y,
    iterations=100000,
    alpha=1e-1
)

prediction = X_norm @ model_w + model_b

plt.figure(figsize=(6,4))
plt.scatter(x, y, color='red', marker='x', label='Actual Value')
plt.plot(x, prediction, label='Predicted Value')
plt.title("Polynomial Regression with Feature Engineering")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()




x = np.arange(20)
y = np.cos(x / 2)

X_poly = np.c_[
    x,
    x**2,
    x**3,
    x**4,
    x**5,
    x**6,
    x**7,
    x**8,
    x**9,
    x**10,
    x**11,
    x**12,
    x**13
]

X_norm, mu, sigma = zscore_normalize_features(X_poly)

model_w, model_b = run_gradient_descent_feng(
    X_norm,
    y,
    iterations=100000,
    alpha=1e-1
)

prediction = X_norm @ model_w + model_b

plt.figure(figsize=(6,4))
plt.scatter(x, y, color='red', marker='x', label='Actual Value')
plt.plot(x, prediction, label='Predicted Value')
plt.title("Cosine Approximation using Polynomial Features")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
