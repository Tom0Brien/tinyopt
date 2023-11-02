import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function


def objective_function(x, y):
    return (x - 1)**2 + (y - 2)**2


# Create a grid of points for plotting
x = np.linspace(0, 2, 400)
y = np.linspace(0, 3, 400)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=5)

# Load the solution from the file
solution = np.loadtxt('solution.dat', skiprows=1)

# Plot the solution
ax.plot([solution[0]], [solution[1]], [objective_function(solution[0], solution[1])],
        'r*', markersize=10, label='Optimal Solution')

# Add labels, legend and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('3D Plot of 2D Quadratic Function Optimization')
ax.legend()

plt.show()
