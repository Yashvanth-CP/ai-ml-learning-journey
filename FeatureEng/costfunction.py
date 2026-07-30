import numpy as np 
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([100.0, 200.0, 300.0, 11.0, 45.0])


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]

    cost_sum =0

    for i in range(m):
        f_wb = w* x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
        total_cost = (1/(2*m)) * cost_sum
    return total_cost
    

# w = float(input(print("enter the valur for w : ")))
# b = float(input(print("enter the valur for b : ")))

# y_pred = w*x_train + b


# cost = compute_cost(x_train, y_train, w, b)
# print("w =", w)
# print("b =", b)
# print("Cost =", cost)

# # Plot actual data
# plt.scatter(x_train, y_train, marker='x', color='red', label='Training Data')

# # Plot prediction line
# plt.plot(x_train, y_pred, label='Prediction Line')

# # Labels
# plt.xlabel("x (Input)")
# plt.ylabel("y (Output)")
# plt.title("Linear Regression")

# plt.legend()

# # Show graph
# plt.show()

# print("cost : ",cost)



# Create values for w and b
w_values = np.linspace(-100, 200, 100)
b_values = np.linspace(-100, 500, 100)

# Create a grid of w and b values
W, B = np.meshgrid(w_values, b_values)

# Create an empty matrix for cost
J = np.zeros(W.shape)

# Calculate cost for every combination of w and b
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J[i, j] = compute_cost(
            x_train,
            y_train,
            W[i, j],
            B[i, j]
        )

# Create 3D figure
fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection='3d')

# Plot the cost surface
surface = ax.plot_surface(
    W,
    B,
    J,
    cmap='viridis',
    alpha=0.8
)

# Labels
ax.set_xlabel('w (Weight)')
ax.set_ylabel('b (Bias)')
ax.set_zlabel('J(w,b) Cost')

ax.set_title('Cost Function J(w,b)')

# Add color bar
fig.colorbar(surface, ax=ax, shrink=0.5)

# Show graph
plt.show()