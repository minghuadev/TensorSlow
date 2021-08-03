import matplotlib.pyplot as plt
import numpy as np
import tensorslow as ts

# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2*np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2*np.ones((50, 2))

# Create a new graph
ts.Graph().as_default()

X = ts.placeholder()
c = ts.placeholder()

# Initialize weights randomly
W = ts.Variable(np.random.randn(2, 2))
b = ts.Variable(np.random.randn(2))

# Build perceptron
p = ts.softmax(ts.add(ts.matmul(X, W), b))

# Build cross-entropy loss
J = ts.negative(ts.reduce_sum(ts.reduce_sum(ts.multiply(c, ts.log(p)), axis=1)))

# Build minimization op
minimization_op = ts.train.GradientDescentOptimizer(learning_rate=0.01).minimize(J)

# Build placeholder inputs
feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    c:
        [[1, 0]] * len(blue_points)
        + [[0, 1]] * len(red_points)

}

# Create session
session = ts.Session()

# Perform 100 gradient descent steps
trained_rec = [] # track training process
for step in range(100):
    J_value = session.run(J, feed_dict)
    if step % 10 == 0:
        print("Step:", step, " Loss:", J_value)
        trained_rec.append([session.run(W).copy(), session.run(b).copy()])
    session.run(minimization_op, feed_dict)

trained_plot = True # set to True to plot the training progress
if trained_plot:
    def plot_hist(w_h, b_h):
        x_ax = np.linspace(-4, 4, 100)
        y_ax = -w_h[0][0] / w_h[1][0] * x_ax - b_h[0] / w_h[1][0]
        # contain to (-4,4) for y_ax:
        i0, i100 = 0, 100
        if y_ax[0] < -4 and y_ax[-1] >= -4:
            for i in range(99):
                if y_ax[i+1] >= -4:
                    i0 = i
                    break
        elif y_ax[0] > 4 and y_ax[-1] <= 4:
            for i in range(99):
                if y_ax[i+1] <= 4:
                    i0 = i
                    break
        if y_ax[99] < -4 and y_ax[i0] >= -4:
            for i in range(99, i0, -1):
                if y_ax[i-1] >= -4:
                    i100 = i+1
                    break
        elif y_ax[99] > 4 and y_ax[i0] <= 4:
            for i in range(99, i0, -1):
                if y_ax[i-1] <= 4:
                    i100 = i+1
                    break
        x_ax = x_ax[i0:i100]
        y_ax = y_ax[i0:i100]

        plt.plot(x_ax, y_ax, color='yellow', zorder=1)
    for hist in trained_rec:
        plot_hist(hist[0], hist[1])

# Print final result
W_value = session.run(W)
print("Weight matrix:\n", W_value)
b_value = session.run(b)
print("Bias:\n", b_value)

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -W_value[0][0]/W_value[1][0] * x_axis - b_value[0]/W_value[1][0]
plt.plot(x_axis, y_axis)
if trained_plot:
    plt.plot(x_axis, y_axis, color='xkcd:cloudy blue')

# Add the red and blue points
plt.scatter(red_points[:,0], red_points[:,1], color='red', zorder=2)
plt.scatter(blue_points[:,0], blue_points[:,1], color='blue', zorder=2)
plt.show()

