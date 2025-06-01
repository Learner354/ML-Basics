import numpy as np
import matplotlib.pyplot as plt
class LinearRegression:
  def __init__(self,learning_rate,iterations):
    self.iterations=iterations
    self.lr=learning_rate
    self.weight=[]
    self.bias=0
    self.loss_history=[]
  def fit(self,X,y):
    n_rows, n_columns=X.shape
    self.weight=np.zeros(n_columns)
 # Gradient descent
    for _ in range(self.iterations):
      y_pred = np.dot(X, self.weight) + self.bias
      dw = (1/n_rows) * np.dot(X.T, (y_pred - y))  # Derivative of MSE w.r.t weights
      db = (1/n_rows) * np.sum(y_pred - y)         # Derivative w.r.t bias
      self.weight -= self.lr * dw
      self.bias -= self.lr * db
      loss = np.mean((y_pred-y)**2)
      self.loss_history.append(loss)

  def predict(self, X):
    return np.dot(X, self.weight) + self.bias
  def plot_loss(self):
    plt.plot(self.loss_history)
    plt.xlabel('iterations')
    plt.ylabel('MSE loss')
    plt.title('Loss curve')
    plt.grid(True)
    plt.show()
##Hyperparameters tuning
def tune_hyperparameters(X, y, learning_rates, iterations):
  best_model = None
  best_loss = float('inf')
  for lrs in learning_rates:
    #for iterations initeration_list:
    model = LinearRegression(lrs,iterations)
    model.fit(X,y)
    print(model.predict([[4]]))  # Should output ~4.0
    final_loss = model.loss_history[-1]
    print(f"LR: {lrs}, Iter: {iter}, Final Loss: {final_loss}")
    if final_loss < best_loss:
      best_loss = final_loss
      best_model = model

X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])
learning_rates = [0.01,0.03, 0.05,0.09, 0.1]
#can use iteration list but then u have to uncomment iters in iteration list and input the list to function
tune_hyperparameters(X, y, learning_rates, iterations=1000)


  


  
