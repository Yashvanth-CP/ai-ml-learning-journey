import math
import numpy as np
import matplotlib.pyplot as plt

from lab_utils_uni import (
    plt_contour_wgrad,
    plt_divergence
)

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])



def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        # Calculate the prediction
        f_wb = w * x[i] + b

        # Calculate squared error
        cost = cost + (f_wb - y[i]) ** 2

    # Calculate the final cost
    total_cost = cost / (2 * m)

    return total_cost

def compute_gradient(x, y, w, b):

    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):

        # Calculate the prediction
        f_wb = w * x[i] + b

        # Calculate the error
        error = f_wb - y[i]

        # Calculate gradient for w
        dj_dw_i = error * x[i]

        # Calculate gradient for b
        dj_db_i = error

        # Add gradients
        dj_dw = dj_dw + dj_dw_i
        dj_db = dj_db + dj_db_i

    # Average the gradients
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db



def gradient_descent(x, y, w_in, b_in, alpha, num_iters):

    # Lists to store cost and parameter values
    J_history = []
    p_history = []

    # Start with initial values
    w = w_in
    b = b_in

    for i in range(num_iters):

        # Calculate gradients
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update w and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Calculate and store the cost
        cost = compute_cost(x, y, w, b)

        J_history.append(cost)
        p_history.append([w, b])

        # Print progress
        if i % math.ceil(num_iters / 10) == 0:
            print(
                f"Iteration {i:4}: "
                f"Cost {cost:0.2e} "
                f"dj_dw: {dj_dw:0.3e} "
                f"dj_db: {dj_db:0.3e} "
                f"w: {w:0.3e} "
                f"b: {b:0.3e}"
            )

    return w, b, J_history, p_history



# Initial values
w_init = 0
b_init = 0

# Gradient descent settings
iterations = 10000
learning_rate = 0.01


# Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    learning_rate,
    iterations
)


# Print the final values of w and b
print()
print(f"Final values found by gradient descent:")
print(f"w = {w_final:.4f}")
print(f"b = {b_final:.4f}")



fig, (ax1, ax2) = plt.subplots(
    1, 2,
    constrained_layout=True,
    figsize=(12, 4)
)

# Show the first 100 iterations
ax1.plot(J_hist[:100])
ax1.set_title("Cost vs. Iteration (Start)")
ax1.set_ylabel("Cost")
ax1.set_xlabel("Iteration")

# Show the cost after 1000 iterations
ax2.plot(
    1000 + np.arange(len(J_hist[1000:])),
    J_hist[1000:]
)

ax2.set_title("Cost vs. Iteration (End)")
ax2.set_ylabel("Cost")
ax2.set_xlabel("Iteration")

plt.show()



prediction_1000 = w_final * 1.0 + b_final
prediction_1200 = w_final * 1.2 + b_final
prediction_2000 = w_final * 2.0 + b_final

print()
print(
    f"1000 sqft house prediction: "
    f"${prediction_1000:.1f} thousand"
)

print(
    f"1200 sqft house prediction: "
    f"${prediction_1200:.1f} thousand"
)

print(
    f"2000 sqft house prediction: "
    f"${prediction_2000:.1f} thousand"
)


fig, ax = plt.subplots(
    1,
    1,
    figsize=(12, 6)
)

plt_contour_wgrad(
    x_train,
    y_train,
    p_hist,
    ax
)

plt.show()


fig, ax = plt.subplots(
    1,
    1,
    figsize=(12, 4)
)

plt_contour_wgrad(
    x_train,
    y_train,
    p_hist,
    ax,
    w_range=[180, 220, 0.5],
    b_range=[80, 120, 0.5],
    contours=[1, 5, 10, 20],
    resolution=0.5
)

plt.show()



# Reset initial values
w_init = 0
b_init = 0

# Use a large learning rate
iterations = 10
large_learning_rate = 0.8


# Run gradient descent again
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    large_learning_rate,
    iterations
)



plt_divergence(
    p_hist,
    J_hist,
    x_train,
    y_train
)

plt.show()