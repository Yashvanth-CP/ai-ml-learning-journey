import numpy as np
import matplotlib.pyplot as plt

from lab_utils_multi import load_house_data, run_gradient_descent
from lab_utils_multi import norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc


# ============================================================
# 1. Load the dataset
# ============================================================

X_train, y_train = load_house_data()

X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']


# ============================================================
# 2. Plot each feature against house price
# ============================================================

fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("Price (1000's)")
plt.show()


# ============================================================
# 3. Try gradient descent without normalization
# ============================================================

# Learning rate = 9.9e-7
_, _, hist = run_gradient_descent(
    X_train,
    y_train,
    10,
    alpha=9.9e-7
)

plot_cost_i_w(X_train, y_train, hist)


# Learning rate = 9e-7
_, _, hist = run_gradient_descent(
    X_train,
    y_train,
    10,
    alpha=9e-7
)

plot_cost_i_w(X_train, y_train, hist)


# Learning rate = 1e-7
_, _, hist = run_gradient_descent(
    X_train,
    y_train,
    10,
    alpha=1e-7
)

plot_cost_i_w(X_train, y_train, hist)


# ============================================================
# 4. Define Z-score normalization function
# ============================================================

def zscore_normalize_features(X):
    """
    Computes X_norm, z-score normalized by column.

    Args:
        X (ndarray (m,n)): Input data with m examples and n features.

    Returns:
        X_norm (ndarray (m,n)): Normalized input data.
        mu (ndarray (n,)): Mean of each feature.
        sigma (ndarray (n,)): Standard deviation of each feature.
    """

    # Find the mean of each feature/column
    mu = np.mean(X, axis=0)

    # Find the standard deviation of each feature/column
    sigma = np.std(X, axis=0)

    # Normalize each feature
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# ============================================================
# 5. Calculate mean, standard deviation, and normalized data
# ============================================================

mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)

# Subtract mean
X_mean = X_train - mu

# Z-score normalization
X_norm = (X_train - mu) / sigma


# ============================================================
# 6. Compare data before and after normalization
# ============================================================

fig, ax = plt.subplots(1, 3, figsize=(12, 3))

# Original data
ax[0].scatter(X_train[:, 0], X_train[:, 3])
ax[0].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[0].set_title("Unnormalized")
ax[0].axis('equal')


# Mean-centered data
ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
ax[1].set_xlabel(X_features[0])
ax[1].set_ylabel(X_features[3])
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')


# Z-score normalized data
ax[2].scatter(X_norm[:, 0], X_norm[:, 3])
ax[2].set_xlabel(X_features[0])
ax[2].set_ylabel(X_features[3])
ax[2].set_title("Z-score normalized")
ax[2].axis('equal')


plt.tight_layout()
fig.suptitle(
    "Distribution of Features Before, During, and After Normalization",
    y=1.05
)

plt.show()


# ============================================================
# 7. Normalize the original features using the function
# ============================================================

X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)

print(f"X_mu = {X_mu}")
print(f"X_sigma = {X_sigma}")

print(
    f"Peak to Peak range by column in Raw X: "
    f"{np.ptp(X_train, axis=0)}"
)

print(
    f"Peak to Peak range by column in Normalized X: "
    f"{np.ptp(X_norm, axis=0)}"
)


# ============================================================
# 8. Plot feature distributions before normalization
# ============================================================

fig, ax = plt.subplots(1, 4, figsize=(12, 3))

for i in range(len(ax)):
    norm_plot(ax[i], X_train[:, i])
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("Count")

fig.suptitle("Distribution of Features Before Normalization")

plt.show()


# ============================================================
# 9. Plot feature distributions after normalization
# ============================================================

fig, ax = plt.subplots(1, 4, figsize=(12, 3))

for i in range(len(ax)):
    norm_plot(ax[i], X_norm[:, i])
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel("Count")

fig.suptitle("Distribution of Features After Normalization")

plt.show()


# ============================================================
# 10. Run gradient descent using normalized features
# ============================================================

w_norm, b_norm, hist = run_gradient_descent(
    X_norm,
    y_train,
    1000,
    1.0e-1
)


# ============================================================
# 11. Predict target values using normalized features
# ============================================================

m = X_norm.shape[0]

yp = np.zeros(m)

for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm


# ============================================================
# 12. Plot predictions and actual target values
# ============================================================

fig, ax = plt.subplots(
    1,
    4,
    figsize=(12, 3),
    sharey=True
)

for i in range(len(ax)):

    # Actual target values
    ax[i].scatter(
        X_train[:, i],
        y_train,
        label='Target'
    )

    # Feature name
    ax[i].set_xlabel(X_features[i])

    # Predicted values
    ax[i].scatter(
        X_train[:, i],
        yp,
        color=dlc["dlorange"],
        label='Prediction'
    )

ax[0].set_ylabel("Price")
ax[0].legend()

fig.suptitle(
    "Target Versus Prediction Using Z-score Normalized Model"
)

plt.show()


# ============================================================
# 13. Predict the price of a new house
# ============================================================

# House details:
# 1200 sqft
# 3 bedrooms
# 1 floor
# 40 years old

x_house = np.array([1200, 3, 1, 40])


# Normalize the new house features
x_house_norm = (x_house - X_mu) / X_sigma

print("Normalized house features:")
print(x_house_norm)


# Predict house price
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm

print(
    f"Predicted price of a house with "
    f"1200 sqft, 3 bedrooms, 1 floor, 40 years old "
    f"= ${x_house_predict * 1000:,.0f}"
)


# ============================================================
# 14. Compare original and normalized features
# ============================================================

plt_equal_scale(
    X_train,
    X_norm,
    y_train
)