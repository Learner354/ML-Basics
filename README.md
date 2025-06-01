# ML-Basics
# Linear Regression from Scratch in Python 

This project implements Linear Regression **from scratch** using NumPy.  
It includes:
- Gradient Descent optimization
- Loss visualization (MSE)
- Hyperparameter tuning for learning rate and iteration count
- Subplot visualizations of multiple learning rates
- Error handling for exploding gradients / NaN loss


## 🚀 Features

- `LinearRegression` class with `.fit()`, `.predict()`, `.plot_loss()`
- `tune_hyperparameters()` to grid-search LR & iterations
- `plot_losses_for_learning_rates()` for visual comparison
- NaN/overflow protection for stability

## Next Steps
- Extend to Logistic Regression
- Add gradient clipping
- Normalize input data
- Implement early stopping

If you'd like to contribute or extend this project (e.g., add batch/mini-batch support), feel free to fork.

## 🧪 Example Usage
```python
model = LinearRegression(learning_rate=0.01, iterations=500)
model.fit(X, y)
model.plot_loss()
print(model.predict([[4]]))

## plot example
plot_losses_for_learning_rates(X, y, [0.001, 0.01, 0.1], iterations=1000)



