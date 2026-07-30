import numpy as np 
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([300,500,700,900,1100,1300,1500])



print(f" x_train = {x_train}")
print(f" Y_train = {y_train}")

print(f"x_train.shape : {x_train.shape}")
m= len(x_train)
print(f"Number of training examples is: {m}")

i = 0 # change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]

print(f"(x^({i}), y^({i})) =  ({x_i}, {y_i})")

#plot the graph 
plt.scatter(x_train, y_train, marker='x', c='g')

#settle the title

plt.title("housing prices")


#set the y-axis label

plt.ylabel('price (in 1000s of dollars)')

#plot x- axis

plt.xlabel('size (1000 sqrt)')

plt.show()

w = 5
b = 150

print(f"w :{ w}")
print(f"b: {b}")


def compute_model_output(x, w, b):

    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i]= w * x[i] + b

    return f_wb


tmp_f_wb = compute_model_output(x_train, w, b)
indices = np.argsort(x_train)

x_sorted = x_train[indices]
y_sorted = tmp_f_wb[indices]

plt.plot(x_sorted, y_sorted, color='blue', label='Prediction')
#plot model prediction 
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
w = 200                         
b = 100    
x_i = 13
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")