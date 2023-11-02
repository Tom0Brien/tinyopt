import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rosenbrock function


def rosenbrock_function(x, y, a=1, b=100):
    return (a - x)**2 + b*(y - x**2)**2


# Create a grid of points for plotting
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function(X, Y)

# Set up the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=5)

# Load the solution from the file
solution = np.loadtxt('rosenbrock_solution.dat', skiprows=1)

# Plot the solution
ax.plot([solution[0]], [solution[1]], [rosenbrock_function(solution[0], solution[1])],
        'r*', markersize=10, label='Optimal Solution')

# Add labels, legend and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('3D Plot of Rosenbrock Function Optimization')
ax.legend()

plt.show()
