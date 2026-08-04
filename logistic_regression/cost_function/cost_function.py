import numpy as np
import matplotlib.pyplot as plt

from lab_utils_common import plot_data, sigmoid, dlc


X_train = np.array([
    [0.5, 1.5],
    [1, 1],
    [1.5, 0.5],
    [3, 0.5],
    [2, 2],
    [1, 2.5]
])

y_train = np.array([0, 0, 0, 1, 1, 1])


# -----------------------------
# Plot training data
# -----------------------------

fig, ax = plt.subplots(figsize=(5, 5))

plot_data(X_train, y_train, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")
ax.set_title("Training Data")
ax.legend()

plt.show()


# -----------------------------
# Logistic Cost Function
# -----------------------------

def compute_cost_logistic(X, y, w, b):

    m = X.shape[0]
    cost = 0

    for i in range(m):

        z_i = np.dot(X[i], w) + b

        f_wb_i = sigmoid(z_i)

        cost += (
            -y[i] * np.log(f_wb_i)
            - (1 - y[i]) * np.log(1 - f_wb_i)
        )

    cost = cost / m

    return cost


# -----------------------------
# Calculate cost
# -----------------------------

w_tmp = np.array([1, 1])
b_tmp = -3

cost = compute_cost_logistic(
    X_train,
    y_train,
    w_tmp,
    b_tmp
)

print("Cost =", cost)


# -----------------------------
# Decision Boundaries
# -----------------------------

x0 = np.arange(0, 6)

x1 = 3 - x0
x1_other = 4 - x0


fig, ax = plt.subplots(figsize=(5, 5))


# Boundary b = -3
ax.plot(
    x0,
    x1,
    color=dlc["dlblue"],
    label="b = -3"
)


# Boundary b = -4
ax.plot(
    x0,
    x1_other,
    color=dlc["dlmagenta"],
    label="b = -4"
)


# Training data
plot_data(X_train, y_train, ax)


ax.axis([0, 4, 0, 4])

ax.set_xlabel("$x_0$")
ax.set_ylabel("$x_1$")

ax.set_title("Decision Boundary")

ax.legend()

plt.show()


# -----------------------------
# Compare Costs
# -----------------------------

w_array1 = np.array([1, 1])
b_1 = -3

w_array2 = np.array([1, 1])
b_2 = -4


cost1 = compute_cost_logistic(
    X_train,
    y_train,
    w_array1,
    b_1
)

cost2 = compute_cost_logistic(
    X_train,
    y_train,
    w_array2,
    b_2
)


print("Cost for b = -3 :", cost1)

print("Cost for b = -4 :", cost2)