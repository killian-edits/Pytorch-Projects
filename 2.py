import numpy as np

X = np.array([14, 18, 22, 26, 30])
y = np.array([20, 30, 38, 50, 62])

w = 2
b = 1

print("Target:", y)

print()

print("Initial weight and bias:", w, b)
print("Initial Guess:", w * X + b)

print()

epochs = 100000

for epoch in range(epochs):
    y_hat = w * X + b

    loss = y_hat - y
    final_loss = np.mean(loss ** 2)

    loss_gradient = 2 * np.mean(loss * X)
    bias_gradient = 2 * np.mean(loss)

    learning_rate = 0.0015
    w -= learning_rate * loss_gradient
    b -= learning_rate * bias_gradient

print("Updated weight and bias:", round(w, 5), round(b, 5))
print("Final Guess:", y_hat)
