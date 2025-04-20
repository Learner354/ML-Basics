import numpy as np
class LinearRegression:
  def __init__(self,learning_rate,iterations):
    self.iterations=iterations
    self.lr=learning_rate
    self.weight=[]
    self.bias=0
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
  def predict(self, X):
    return np.dot(X, self.weight) + self.bias
  

X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])
model = LinearRegression(0.01,200)
model.fit(X, y)
print(model.predict([[4]]))  # Should output ~4.0

  


  
