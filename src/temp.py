import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define combined function: exponential + logarithmic
def combined_model(t, a, b, c, d):
    return a * np.exp(b * t) + c * np.log(t + d)

# Example time points (e.g., months)
t = np.linspace(1, 20, 100)

# Simulated data for fitting (replace with actual poll data)
y_data = 100 * np.exp(0.3 * t) + 50 * np.log(t + 1) + np.random.normal(0, 20, len(t))

# Initial guess for parameters (a, b, c, d)
initial_guess = [100, 0.3, 50, 1]

# Curve fitting
params, covariance = curve_fit(combined_model, t, y_data, p0=initial_guess)

# Predicted values using the fitted parameters
y_fit = combined_model(t, *params)

# Plotting
plt.scatter(t, y_data, label="Poll Data")
plt.plot(t, y_fit, color='red', label="Fitted Model")
plt.title('Vote Opinion Poll - Exponential and Logarithmic Combined Model')
plt.xlabel('Time (Polling Period)')
plt.ylabel('Voter Support (%)')
plt.legend()
plt.show()
